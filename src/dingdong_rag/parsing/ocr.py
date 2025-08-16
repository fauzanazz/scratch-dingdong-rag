from __future__ import annotations

import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover
    genai = None
    GEMINI_AVAILABLE = False


@dataclass
class GeminiConfig:
    model: str = "gemini-2.0-flash"  # Updated to use the higher-accuracy model
    api_key: Optional[str] = None
    max_retries: int = 2
    timeout_seconds: int = 30
    temperature: float = 0.0
    safety_settings: Optional[Dict] = None
    daily_limit: int = 1500
    usage_dir: Optional[str] = None
    estimated_cost_per_page_usd: float = 0.00016  # ~6,000 pages per dollar as per blog
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    ocr_mode: str = "markdown"  # New: markdown, text, or markdown_chunked
    chunk_size_words: Tuple[int, int] = (250, 1000)  # Min and max words per chunk
    enable_chunking: bool = False  # New: enable combined OCR+chunking


def extract_single_page_ocr(pdf_path: str, page_num: int, dpi: int = 300, lang: str = "eng") -> str:
    if fitz is None:
        # Fallback path using pdf2image
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)
        if not pages:
            return ""
        page = pages[0]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            page.save(temp_file.name, "PNG")
            try:
                text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)
                return text.strip() if text else ""
            finally:
                os.unlink(temp_file.name)

    doc = fitz.open(pdf_path)
    if page_num >= doc.page_count:
        doc.close()
        raise ValueError(f"Page {page_num} does not exist (PDF has {doc.page_count} pages)")

    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        pix.save(temp_file.name)
        pix = None
        try:
            text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)
            return text.strip() if text else ""
        finally:
            os.unlink(temp_file.name)


def extract_single_page_ocr_gemini(pdf_path: str, page_num: int, config: GeminiConfig | None = None) -> str:
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini not available. Please install with: uv add google-generativeai")

    if config is None:
        config = GeminiConfig()

    # Configure Gemini API
    api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY or pass api_key in GeminiConfig")
    genai.configure(api_key=api_key)

    # Render page to image
    if fitz is None:
        pages = convert_from_path(pdf_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)
        if not pages:
            return ""
        img_bytes_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pages[0].save(img_bytes_tmp.name, "PNG")
        img = Image.open(img_bytes_tmp.name)
        os.unlink(img_bytes_tmp.name)
    else:
        doc = fitz.open(pdf_path)
        if page_num >= doc.page_count:
            raise ValueError(f"Page {page_num} does not exist in PDF (total pages: {doc.page_count})")
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        import io

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()

    # Build request based on OCR mode
    if config.ocr_mode == "markdown_chunked" and config.enable_chunking:
        prompt = _get_chunking_prompt(config.chunk_size_words)
    elif config.ocr_mode == "markdown":
        prompt = _get_markdown_prompt()
    else:
        # Legacy text extraction prompt
        prompt = (
            "Please extract all text from this image exactly as it appears. "
            "Preserve the original formatting, layout, and structure as much as possible. "
            "Include all visible text including headers, body text, captions, tables, etc. "
            "Do not add any commentary or explanations - just return the extracted text."
        )

    for attempt in range(config.max_retries):
        try:
            response = genai.GenerativeModel(config.model).generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=4096,
                ),
                safety_settings=config.safety_settings or [],
            )
            if response.text:
                return response.text.strip()
        except Exception:
            if attempt == config.max_retries - 1:
                raise
            sleep_time = min(config.backoff_base_seconds * (2 ** attempt), config.backoff_max_seconds)
            sleep_time += random.uniform(0.0, 0.5)
            time.sleep(sleep_time)

    return ""


def get_physical_cpu_cores() -> int:
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores and cores > 0:
            return cores
    except Exception:
        pass
    try:
        logical = cpu_count()
        if logical > 2 and logical % 2 == 0 and logical <= 16:
            return logical // 2
        return logical
    except Exception:
        return 4


def extract_multiple_pages_ocr_parallel(
    pdf_path: str,
    page_numbers: List[int],
    dpi: int = 300,
    lang: str = "eng",
    max_workers: Optional[int] = None,
    ocr_method: str = "tesseract",
    gemini_config: GeminiConfig | None = None,
) -> Dict[int, str]:
    if not page_numbers:
        return {}

    if ocr_method.lower() == "gemini":
        max_workers = min(len(page_numbers), (max_workers or 3), 3)
    else:
        if max_workers is None:
            max_workers = min(len(page_numbers), get_physical_cpu_cores())

    results: Dict[int, str] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _extract_single_page_worker, (pdf_path, page_num, dpi, lang, ocr_method, gemini_config)
            ): page_num
            for page_num in page_numbers
        }
        for future in as_completed(futures):
            page_num, text = future.result()
            results[page_num] = text

    return results


def _extract_single_page_worker(args: Tuple[str, int, int, str, str, Optional[GeminiConfig]]) -> Tuple[int, str]:
    pdf_path, page_num, dpi, lang, ocr_method, gemini_config = args
    try:
        if ocr_method.lower() == "gemini":
            if not GEMINI_AVAILABLE:
                text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
            else:
                text = extract_single_page_ocr_gemini(pdf_path, page_num, gemini_config)
        else:
            text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
        return page_num, text
    except Exception:
        if ocr_method.lower() == "gemini":
            try:
                text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
                return page_num, text
            except Exception:
                pass
        return page_num, ""


def _get_markdown_prompt() -> str:
    """Get the high-performance markdown OCR prompt based on the blog post."""
    return (
        "OCR the following page into Markdown. Tables should be formatted as HTML. "
        "Do not surround your output with triple backticks."
    )


def _get_chunking_prompt(chunk_size_words: Tuple[int, int] = (250, 1000)) -> str:
    """Get the combined OCR+chunking prompt for cost-effective processing."""
    min_words, max_words = chunk_size_words
    return f"""\
OCR the following page into Markdown. Tables should be formatted as HTML. 
Do not sorround your output with triple backticks.

Chunk the document into sections of roughly {min_words} - {max_words} words. Our goal is 
to identify parts of the page with same semantic theme. These chunks will 
be embedded and used in a RAG pipeline. 

Surround the chunks with <chunk> </chunk> html tags.
"""


def extract_text_with_gemini_chunking(pdf_path: str, config: GeminiConfig | None = None) -> List[str]:
    """Extract and chunk text from a PDF using Gemini in a single pass for maximum cost efficiency.
    
    Args:
        pdf_path: Path to the PDF file
        config: GeminiConfig with chunking enabled
        
    Returns:
        List of chunked text segments
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini not available. Please install with: uv add google-generativeai")
    
    if config is None:
        config = GeminiConfig(enable_chunking=True, ocr_mode="markdown_chunked")
    
    # Process all pages and collect chunked content
    chunks = []
    
    # Get page count
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) required for page counting")
    
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()
    
    # Process pages in batches to manage API costs
    for page_num in range(total_pages):
        try:
            page_content = extract_single_page_ocr_gemini(pdf_path, page_num, config)
            if page_content.strip():
                page_chunks = _extract_chunks_from_content(page_content)
                chunks.extend(page_chunks)
        except Exception as e:
            print(f"Warning: Failed to process page {page_num}: {e}")
            continue
    
    return chunks


def _extract_chunks_from_content(content: str) -> List[str]:
    """Extract individual chunks from OCR content that contains <chunk> tags."""
    import re
    
    # Find all content between <chunk> and </chunk> tags
    chunk_pattern = r'<chunk>(.*?)</chunk>'
    matches = re.findall(chunk_pattern, content, re.DOTALL | re.IGNORECASE)
    
    # Clean up chunks
    chunks = []
    for match in matches:
        cleaned = match.strip()
        if cleaned:
            chunks.append(cleaned)
    
    # If no chunk tags found, return the entire content as one chunk
    if not chunks and content.strip():
        chunks.append(content.strip())
    
    return chunks
