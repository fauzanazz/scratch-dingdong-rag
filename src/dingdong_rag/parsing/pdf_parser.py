# Load environment variables first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Silently fail if dotenv is not available

import argparse
import os
import re
import time
from pathlib import Path
import pymupdf4llm
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import gc
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import psutil
import random

from .validation import (
    ValidationConfig,
    ValidationResult,
    validate_extracted_text,
    validate_per_page_content,
    split_text_by_pages,
)
from .docling_adapter import (
    DoclingConfig,
    extract_text_with_docling,
    DOCLING_AVAILABLE,
)
from .ocr import (
    GeminiConfig,
    extract_single_page_ocr,
    extract_single_page_ocr_gemini,
    extract_multiple_pages_ocr_parallel,
    GEMINI_AVAILABLE,
)

# Import genai conditionally
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from .hybrid import extract_text_hybrid
from .enhanced_pymupdf import (
    extract_with_enhanced_pymupdf,
    EnhancedPyMuPDFConfig,
    format_enhanced_output,
    PYMUPDF_AVAILABLE as ENHANCED_PYMUPDF_AVAILABLE
)

class _LoggerShim:
    def info(self, fmt: str, *args):
        try:
            print(str(fmt).format(*args))
        except Exception:
            print(str(fmt))

    def warning(self, fmt: str, *args):
        try:
            print("WARNING: " + str(fmt).format(*args))
        except Exception:
            print("WARNING: " + str(fmt))

log = _LoggerShim()

_OCR_RENDERER = os.getenv("DD_RAG_OCR_RENDERER", "pymupdf").lower()
_USE_FITZ_RENDERER = _OCR_RENDERER != "pdf2image"

# Usage tracking (optional)
try:
    from ..api.api_usage import get_gemini_usage_tracker
except Exception:
    get_gemini_usage_tracker = None  # Tracker optional

def get_physical_cpu_cores() -> int:
    """Get the number of physical CPU cores (not logical processors/threads)."""
    try:
        # Try to get physical cores using psutil
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores and physical_cores > 0:
            return physical_cores
    except (ImportError, AttributeError):
        pass
    
    try:
        # Fallback: Try to detect hyperthreading and divide by 2
        logical_cores = cpu_count()
        
        # Simple heuristic: if logical cores is even and > 2, likely has hyperthreading
        if logical_cores > 2 and logical_cores % 2 == 0:
            # Check if it's likely hyperthreading (common pattern)
            if logical_cores <= 16:  # Most consumer CPUs
                return logical_cores // 2
        
        # If we can't determine, use logical count but warn
        return logical_cores
        
    except Exception:
        # Ultimate fallback
        return 4


@dataclass
class ValidationConfig:
    """Configuration for text extraction validation."""
    min_chars_per_page: int = 20
    min_word_ratio: float = 0.3  # Minimum ratio of actual words to total tokens
    max_symbol_ratio: float = 0.3  # Maximum ratio of symbols/special chars
    min_alpha_ratio: float = 0.5  # Minimum ratio of alphabetic characters

@dataclass
class ValidationResult:
    """Result of text validation check."""
    is_valid: bool
    char_count: int
    word_ratio: float
    symbol_ratio: float
    alpha_ratio: float
    reason: str

# DoclingConfig and extract_text_with_docling are now imported from docling_adapter

@dataclass
class GeminiConfig:
    """Configuration for Gemini OCR."""
    model: str = "gemini-2.0-flash-lite"
    api_key: Optional[str] = None  # If None, will try to get from environment
    max_retries: int = 2
    timeout_seconds: int = 30
    temperature: float = 0.0
    safety_settings: Optional[Dict] = None
    # Usage tracking and limits
    daily_limit: int = 1500  # 0 disables limit enforcement
    usage_dir: Optional[str] = None  # Directory for usage logs; defaults to .cache/api_usage
    estimated_cost_per_page_usd: float = 0.0
    # Retry/backoff tuning
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0

def extract_single_page_ocr_gemini(pdf_path: str, page_num: int, config: GeminiConfig = None) -> str:
    """
    Extract text from a single PDF page using Gemini Vision OCR.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        config: Gemini configuration
        
    Returns:
        Extracted text from the page
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini not available. Please install with: uv add google-generativeai")
    
    if config is None:
        config = GeminiConfig()
    
    try:
        # Initialize usage tracker (optional)
        tracker = None
        if get_gemini_usage_tracker is not None:
            if config and config.usage_dir:
                # Allow runtime override for this process
                os.environ["GEMINI_USAGE_DIR"] = str(config.usage_dir)
            tracker = get_gemini_usage_tracker()

        # Configure Gemini API
        if not GEMINI_AVAILABLE or genai is None:
            raise ImportError("Gemini not available. Please install with: uv add google-generativeai")
            
        api_key = config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key in GeminiConfig"
            )
        
        genai.configure(api_key=api_key)
        
        # Convert PDF page to image using PyMuPDF
        import fitz
        doc = fitz.open(pdf_path)
        if page_num >= doc.page_count:
            raise ValueError(f"Page {page_num} does not exist in PDF (total pages: {doc.page_count})")
        
        page = doc[page_num]
        # Use higher DPI for better OCR quality
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        doc.close()
        pix = None  # Free memory
        
        # Prepare image for Gemini
        import io
        from PIL import Image as PILImage
        
        # Convert bytes to PIL Image for consistency
        img = PILImage.open(io.BytesIO(img_bytes))
        
        # Initialize Gemini model
        model = genai.GenerativeModel(config.model)
        
        # Safety settings to prevent blocking
        safety_settings = config.safety_settings or [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        
        # OCR prompt
        prompt = """Please extract all text from this image exactly as it appears. 
        Preserve the original formatting, layout, and structure as much as possible.
        Include all visible text including headers, body text, captions, tables, etc.
        Do not add any commentary or explanations - just return the extracted text."""
        
        # Generate text with retries
        for attempt in range(config.max_retries):
            # Optional daily limit enforcement
            if tracker is not None and (config.daily_limit or 0) > 0:
                if not tracker.can_make_request(config.daily_limit):
                    raise RuntimeError("Gemini daily request limit reached; consider enabling Tesseract fallback or reset tomorrow")
            # Record the attempt and cost estimate (per API call/page)
            if tracker is not None:
                try:
                    tracker.record_attempt(pages=1, estimated_cost_usd=getattr(config, 'estimated_cost_per_page_usd', 0.0))
                except Exception:
                    pass
            try:
                response = model.generate_content(
                    [prompt, img],
                    generation_config=genai.types.GenerationConfig(
                        temperature=config.temperature,
                        max_output_tokens=4096,
                    ),
                    safety_settings=safety_settings
                )
                
                if response.text:
                    if tracker is not None:
                        try:
                            tracker.record_success()
                        except Exception:
                            pass
                    return response.text.strip()
                else:
                    log.warning("Gemini returned empty response for page {} attempt {}", page_num + 1, attempt + 1)
                    
            except Exception as e:
                log.warning("Gemini API error for page {} attempt {}: {}", page_num + 1, attempt + 1, e)
                if tracker is not None:
                    try:
                        tracker.record_failure()
                    except Exception:
                        pass
                if attempt == config.max_retries - 1:
                    raise
                # Exponential backoff with jitter
                sleep_time = min(config.backoff_base_seconds * (2 ** attempt), config.backoff_max_seconds)
                sleep_time += random.uniform(0.0, 0.5)
                time.sleep(sleep_time)
        
        return ""  # Fallback if all attempts fail
        
    except Exception as e:
        raise RuntimeError(f"Gemini OCR failed for page {page_num + 1}: {e}")

def validate_extracted_text(text: str, config: ValidationConfig = None) -> ValidationResult:
    """Validate quality of extracted text to determine if OCR is needed."""
    if config is None:
        config = ValidationConfig()
    
    if not text or not text.strip():
        return ValidationResult(
            is_valid=False, char_count=0, word_ratio=0, symbol_ratio=0, alpha_ratio=0,
            reason="Empty or whitespace-only text"
        )
    
    clean_text = text.strip()
    char_count = len(clean_text)
    
    # Check minimum character count
    if char_count < config.min_chars_per_page:
        return ValidationResult(
            is_valid=False, char_count=char_count, word_ratio=0, symbol_ratio=0, alpha_ratio=0,
            reason=f"Insufficient characters ({char_count} < {config.min_chars_per_page})"
        )
    
    # Split into tokens for analysis
    tokens = re.findall(r'\S+', clean_text)
    if not tokens:
        return ValidationResult(
            is_valid=False, char_count=char_count, word_ratio=0, symbol_ratio=0, alpha_ratio=0,
            reason="No tokens found"
        )
    
    # Count different types of characters
    alpha_chars = sum(1 for c in clean_text if c.isalpha())
    digit_chars = sum(1 for c in clean_text if c.isdigit())
    space_chars = sum(1 for c in clean_text if c.isspace())
    symbol_chars = char_count - alpha_chars - digit_chars - space_chars
    
    # Calculate ratios
    alpha_ratio = alpha_chars / char_count if char_count > 0 else 0
    symbol_ratio = symbol_chars / char_count if char_count > 0 else 0
    
    # Count actual words (tokens with mostly alphabetic characters)
    word_count = sum(1 for token in tokens if re.match(r'^[a-zA-Z]{2,}', token))
    word_ratio = word_count / len(tokens) if tokens else 0
    
    # Validation checks
    if alpha_ratio < config.min_alpha_ratio:
        return ValidationResult(
            is_valid=False, char_count=char_count, word_ratio=word_ratio, symbol_ratio=symbol_ratio, alpha_ratio=alpha_ratio,
            reason=f"Too few alphabetic characters ({alpha_ratio:.2f} < {config.min_alpha_ratio})"
        )
    
    if symbol_ratio > config.max_symbol_ratio:
        return ValidationResult(
            is_valid=False, char_count=char_count, word_ratio=word_ratio, symbol_ratio=symbol_ratio, alpha_ratio=alpha_ratio,
            reason=f"Too many symbols/special characters ({symbol_ratio:.2f} > {config.max_symbol_ratio})"
        )
    
    if word_ratio < config.min_word_ratio:
        return ValidationResult(
            is_valid=False, char_count=char_count, word_ratio=word_ratio, symbol_ratio=symbol_ratio, alpha_ratio=alpha_ratio,
            reason=f"Too few recognizable words ({word_ratio:.2f} < {config.min_word_ratio})"
        )
    
    return ValidationResult(
        is_valid=True, char_count=char_count, word_ratio=word_ratio, symbol_ratio=symbol_ratio, alpha_ratio=alpha_ratio,
        reason="Text quality acceptable"
    )

def validate_per_page_content(content: str, config: ValidationConfig = None) -> Tuple[bool, str]:
    """Validate content by checking each page separately."""
    if config is None:
        config = ValidationConfig()
    
    # Split content by page markers (common patterns)
    page_patterns = [
        r'\n\s*Page \d+\s*\n',
        r'\n\s*## Page \d+\s*\n',
        r'\n\s*\d+\s*\n\s*\n',
        r'\f'  # Form feed character
    ]
    
    pages = [content]  # Default to treating entire content as one page
    
    for pattern in page_patterns:
        if re.search(pattern, content):
            pages = re.split(pattern, content)
            break
    
    failed_pages = 0
    total_pages = len(pages)
    reasons = []
    
    for i, page_content in enumerate(pages):
        if not page_content.strip():
            continue
            
        result = validate_extracted_text(page_content, config)
        if not result.is_valid:
            failed_pages += 1
            reasons.append(f"Page {i+1}: {result.reason}")
    
    # Consider content invalid if more than 30% of pages fail validation
    failure_threshold = 0.3
    failure_rate = failed_pages / total_pages if total_pages > 0 else 1
    
    if failure_rate > failure_threshold:
        summary = f"Failed validation on {failed_pages}/{total_pages} pages ({failure_rate:.1%})"
        return False, f"{summary}. Issues: {'; '.join(reasons[:3])}"
    
    return True, f"Passed validation ({total_pages - failed_pages}/{total_pages} pages valid)"


def split_text_by_pages(content: str) -> List[str]:
    """Split document content into individual pages."""
    if not content.strip():
        return []
    
    # Try multiple page splitting patterns
    page_patterns = [
        r'\n\s*## Page \d+\s*\n',
        r'\n\s*Page \d+\s*\n', 
        r'\n\s*\d+\s*\n\s*\n',
        r'\f'  # Form feed character
    ]
    
    for pattern in page_patterns:
        if re.search(pattern, content):
            pages = re.split(pattern, content)
            # Filter out empty pages
            return [p.strip() for p in pages if p.strip()]
    
    # If no page markers found, treat as single page
    return [content.strip()]


def extract_single_page_ocr(pdf_path: str, page_num: int, dpi: int = 300, lang: str = 'eng') -> str:
    """Extract text from a single PDF page using OCR.

    Tries PyMuPDF for rasterization first. On any PyMuPDF error (not only ImportError),
    falls back to pdf2image to avoid repeatedly triggering MuPDF parser errors.
    """
    # Attempt using PyMuPDF first (unless forced to pdf2image)
    if _USE_FITZ_RENDERER:
        try:
            import fitz  # type: ignore
            doc = fitz.open(pdf_path)
            try:
                if page_num >= doc.page_count:
                    raise ValueError(f"Page {page_num} does not exist (PDF has {doc.page_count} pages)")
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                pix.save(temp_file.name)
                pix = None
                try:
                    text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)
                    return text.strip() if text else ""
                finally:
                    os.unlink(temp_file.name)
        except ImportError:
            # PyMuPDF not installed; fall through to pdf2image
            pass
        except Exception:
            # Any PyMuPDF runtime error (e.g., syntax error in content stream) → fallback
            pass

    # Fallback to pdf2image rendering
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)
        if not pages:
            return ""
        page_img = pages[0]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            page_img.save(temp_file.name, 'PNG')
            page_img = None
            try:
                text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)
                return text.strip() if text else ""
            finally:
                os.unlink(temp_file.name)
    except Exception as e:
        raise RuntimeError(f"Single page OCR failed for page {page_num}: {e}")


def get_page_count(pdf_path: str) -> int:
    """Get the total number of pages in a PDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except ImportError:
        # Fallback using pdf2image
        try:
            pages = convert_from_path(pdf_path, first_page=1, last_page=1)
            # This is inefficient but works as fallback
            test_pages = convert_from_path(pdf_path)
            return len(test_pages)
        except Exception:
            return 0


def extract_single_page_ocr_worker(args: Tuple[str, int, int, str, str, Optional[GeminiConfig]]) -> Tuple[int, str]:
    """Worker function for parallel OCR processing."""
    pdf_path, page_num, dpi, lang, ocr_method, gemini_config = args
    try:
        if ocr_method.lower() == "gemini":
            if not GEMINI_AVAILABLE:
                log.info("Gemini not available, falling back to Tesseract for page {}", page_num + 1)
                text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
            else:
                text = extract_single_page_ocr_gemini(pdf_path, page_num, gemini_config)
        else:
            # Default to Tesseract
            text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
        return page_num, text
    except Exception as e:
        log.warning("{} OCR failed for page {}: {}", ocr_method.capitalize(), page_num + 1, e)
        # Fallback to Tesseract if Gemini fails
        if ocr_method.lower() == "gemini":
            try:
                log.info("Falling back to Tesseract for page {}", page_num + 1)
                text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
                return page_num, text
            except Exception as e2:
                log.warning("Tesseract fallback also failed for page {}: {}", page_num + 1, e2)
        return page_num, ""


def extract_multiple_pages_ocr_parallel(pdf_path: str, page_numbers: List[int], 
                                       dpi: int = 300, lang: str = 'eng', 
                                       max_workers: Optional[int] = None,
                                       ocr_method: str = "tesseract",
                                       gemini_config: GeminiConfig = None) -> Dict[int, str]:
    """Extract text from multiple PDF pages using parallel OCR processing."""
    if not page_numbers:
        return {}
    
    # For Gemini OCR, use fewer workers to avoid API rate limits
    if ocr_method.lower() == "gemini":
        if max_workers is None:
            max_workers = min(len(page_numbers), 3)  # Conservative limit for API calls
        else:
            max_workers = min(max_workers, 3)  # Cap at 3 for API stability
        log.info("Using {} workers for Gemini OCR (rate limit)", max_workers)
    else:
        if max_workers is None:
            physical_cores = get_physical_cpu_cores()
            logical_cores = cpu_count()
            max_workers = min(len(page_numbers), physical_cores)  # Limit to physical CPU cores
            
            if physical_cores != logical_cores:
                log.info("Detected {} physical cores ({} logical)", physical_cores, logical_cores)
    
    log.info("Starting parallel {} OCR for {} pages using {} workers", ocr_method.upper(), len(page_numbers), max_workers)
    
    # Prepare work arguments
    work_args = [(pdf_path, page_num, dpi, lang, ocr_method, gemini_config) for page_num in page_numbers]
    
    results = {}
    
    # Use ProcessPoolExecutor since OCR is CPU-bound (Tesseract computation)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(extract_single_page_ocr_worker, args): args[1] 
            for args in work_args
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_page):
            page_num, text = future.result()
            results[page_num] = text
            completed += 1
            
            if completed % max(1, len(page_numbers) // 10) == 0:  # Progress updates
                progress = completed / len(page_numbers) * 100
                log.info("OCR progress: {:.0f}% ({}/{})", progress, completed, len(page_numbers))
    
    log.info("Parallel OCR completed: {} successful", len([t for t in results.values() if t.strip()]))
    return results


def extract_full_pdf_ocr_parallel(pdf_path: str, dpi: int = 300, lang: str = 'eng',
                                 max_workers: Optional[int] = None,
                                 ocr_method: str = "tesseract", 
                                 gemini_config: GeminiConfig = None) -> str:
    """Extract text from entire PDF using parallel OCR processing."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        
        if total_pages == 0:
            return ""
            
        log.info("Starting parallel OCR for entire PDF ({} pages)...", total_pages)
        
        # Extract all pages in parallel
        page_numbers = list(range(total_pages))
        page_results = extract_multiple_pages_ocr_parallel(
            pdf_path, page_numbers, dpi, lang, max_workers, ocr_method, gemini_config
        )
        
        # Combine results in page order
        extracted_text = []
        for page_num in range(total_pages):
            text = page_results.get(page_num, "")
            if text.strip():
                extracted_text.append(f"## Page {page_num + 1}\n\n{text.strip()}\n")
        
        return "\n".join(extracted_text)
        
    except ImportError:
        # Fallback to sequential processing if PyMuPDF not available
        log.info("PyMuPDF not available, falling back to sequential OCR...")
        return extract_text_with_ocr(pdf_path, dpi, lang)
    except Exception as e:
        # If PyMuPDF failed for page count or any issue, fallback to sequential OCR
        log.warning("Parallel OCR extraction failed for {}: {}. Falling back to sequential OCR...", pdf_path, e)
        return extract_text_with_ocr(pdf_path, dpi, lang)


def extract_text_hybrid(pdf_path: str, dpi: int = 300, lang: str = 'eng', 
                       validation_config: ValidationConfig = None, 
                       max_workers: Optional[int] = None, 
                       use_parallel_ocr: bool = True,
                       parser_method: str = "pymupdf",
                       docling_config: DoclingConfig = None,
                       ocr_method: str = "tesseract",
                       gemini_config: GeminiConfig = None) -> Tuple[str, Dict[str, any]]:
    """
    Extract text using hybrid approach: Primary parser + selective OCR for poor pages only.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for OCR processing
        lang: Language for OCR (tesseract format)
        validation_config: Configuration for text validation
        max_workers: Max workers for parallel OCR
        use_parallel_ocr: Whether to use parallel OCR processing
        parser_method: Primary parser method ("pymupdf" or "docling")
        docling_config: Configuration for Docling (if using Docling)
        ocr_method: OCR method ("tesseract" or "gemini")
        gemini_config: Configuration for Gemini OCR (if using Gemini)
    
    Returns:
        Tuple of (extracted_text, processing_stats)
    """
    if validation_config is None:
        validation_config = ValidationConfig()
    
    print(f"  → Starting hybrid extraction...")
    
    # Step 1: Extract text using the specified parser
    initial_content = ""
    base_stats = {}
    
    if parser_method.lower() == "docling" and DOCLING_AVAILABLE:
        try:
            print(f"  → Using Docling for initial extraction...")
            initial_content, docling_stats = extract_text_with_docling(pdf_path, docling_config)
            base_stats.update(docling_stats)
            base_stats["primary_parser"] = "docling"
        except Exception as e:
            print(f"  → Docling extraction failed: {e}, falling back to PyMuPDF...")
            try:
                initial_content = pymupdf4llm.to_markdown(pdf_path)
                base_stats["primary_parser"] = "pymupdf_fallback"
            except Exception as e2:
                print(f"  → PyMuPDF extraction also failed: {e2}, using full OCR")
                return extract_text_with_ocr(pdf_path, dpi, lang), {"method": "full_ocr_fallback", "ocr_pages": "all", "primary_parser": "failed"}
    elif parser_method.lower() == "docling" and not DOCLING_AVAILABLE:
        print(f"  → Docling requested but not available, using PyMuPDF...")
        try:
            if ENHANCED_PYMUPDF_AVAILABLE:
                print(f"  → Using enhanced PyMuPDF extraction...")
                enhanced_config = EnhancedPyMuPDFConfig(
                    extract_tables=True,
                    extract_images=True,
                    extract_hyperlinks=True,
                    detect_columns=True,
                    output_format="enhanced"
                )
                structure = extract_with_enhanced_pymupdf(pdf_path, enhanced_config)
                initial_content = format_enhanced_output(structure, "enhanced")
                base_stats["primary_parser"] = "enhanced_pymupdf_fallback"
                base_stats["tables_found"] = len(structure.tables)
                base_stats["images_found"] = len(structure.images)
                base_stats["links_found"] = len(structure.hyperlinks)
            else:
                initial_content = pymupdf4llm.to_markdown(pdf_path)
                base_stats["primary_parser"] = "pymupdf_fallback"
        except Exception as e:
            print(f"  → PyMuPDF extraction failed: {e}, falling back to full OCR")
            return extract_text_with_ocr(pdf_path, dpi, lang), {"method": "full_ocr_fallback", "ocr_pages": "all", "primary_parser": "failed"}
    else:
        # Default to PyMuPDF (enhanced if available)
        try:
            if ENHANCED_PYMUPDF_AVAILABLE:
                print(f"  → Using enhanced PyMuPDF extraction...")
                enhanced_config = EnhancedPyMuPDFConfig(
                    extract_tables=True,
                    extract_images=True,
                    extract_hyperlinks=True,
                    detect_columns=True,
                    output_format="enhanced"
                )
                structure = extract_with_enhanced_pymupdf(pdf_path, enhanced_config)
                initial_content = format_enhanced_output(structure, "enhanced")
                base_stats["primary_parser"] = "enhanced_pymupdf"
                base_stats["tables_found"] = len(structure.tables)
                base_stats["images_found"] = len(structure.images)
                base_stats["links_found"] = len(structure.hyperlinks)
            else:
                initial_content = pymupdf4llm.to_markdown(pdf_path)
                base_stats["primary_parser"] = "pymupdf"
        except Exception as e:
            print(f"  → PyMuPDF extraction failed: {e}, falling back to full OCR")
            return extract_text_with_ocr(pdf_path, dpi, lang), {"method": "full_ocr_fallback", "ocr_pages": "all", "primary_parser": "failed"}
    
    if not initial_content.strip():
        print(f"  → {parser_method.capitalize()} yielded empty content, using full OCR")
        return extract_text_with_ocr(pdf_path, dpi, lang), {"method": "full_ocr_empty", "ocr_pages": "all", **base_stats}
    
    # Step 2: Split content by pages
    pages = split_text_by_pages(initial_content)
    total_pages = len(pages)
    
    if total_pages <= 1:
        # Single page document, validate as whole
        validation = validate_extracted_text(initial_content, validation_config)
        if validation.is_valid:
            print(f"  → Single page validation passed")
            return initial_content, {"method": "single_page_success", "total_pages": 1, "ocr_pages": 0, **base_stats}
        else:
            print(f"  → Single page validation failed: {validation.reason}, using OCR")
            ocr_content = extract_text_with_ocr(pdf_path, dpi, lang)
            return ocr_content, {"method": "single_page_ocr", "total_pages": 1, "ocr_pages": 1, **base_stats}
    
    # Step 3: Validate each page and identify poor ones
    print(f"  → Validating {total_pages} pages individually...")
    poor_pages = []
    page_validations = {}
    
    for i, page_content in enumerate(pages):
        validation = validate_extracted_text(page_content, validation_config)
        page_validations[i] = validation
        
        if not validation.is_valid:
            poor_pages.append(i)
            print(f"    Page {i+1}: Poor quality - {validation.reason}")
    
    # Step 4: Decide on processing approach
    poor_page_ratio = len(poor_pages) / total_pages
    
    if not poor_pages:
        # All pages are good quality
        print(f"  → All pages passed validation, using {base_stats.get('primary_parser', 'primary parser')} result")
        return initial_content, {
            "method": "primary_parser_only", 
            "total_pages": total_pages, 
            "ocr_pages": 0,
            "poor_page_ratio": 0.0,
            **base_stats
        }
    
    elif poor_page_ratio > 0.7:  # More than 70% pages are poor
        print(f"  → {poor_page_ratio:.1%} pages are poor quality, using full OCR")
        if use_parallel_ocr and len(poor_pages) >= 2:  # Use parallel for 2+ pages
            ocr_content = extract_full_pdf_ocr_parallel(pdf_path, dpi, lang, max_workers, ocr_method, gemini_config)
        else:
            if ocr_method.lower() == "gemini" and GEMINI_AVAILABLE:
                # For Gemini, use single-page extraction for non-parallel mode
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    total_pages = doc.page_count
                    doc.close()
                    
                    extracted_pages = []
                    for page_num in range(total_pages):
                        page_text = extract_single_page_ocr_gemini(pdf_path, page_num, gemini_config)
                        if page_text.strip():
                            extracted_pages.append(f"## Page {page_num + 1}\n\n{page_text.strip()}\n")
                    
                    ocr_content = "\n".join(extracted_pages)
                except Exception as e:
                    print(f"  → Gemini sequential OCR failed: {e}, falling back to Tesseract")
                    ocr_content = extract_text_with_ocr(pdf_path, dpi, lang)
            else:
                ocr_content = extract_text_with_ocr(pdf_path, dpi, lang)
        return ocr_content, {
            "method": "full_ocr_threshold", 
            "total_pages": total_pages, 
            "ocr_pages": total_pages,
            "poor_page_ratio": poor_page_ratio,
            "parallel_processing": use_parallel_ocr and len(poor_pages) >= 2,
            "ocr_method": ocr_method,
            **base_stats
        }
    
    else:
        # Selective OCR: only poor pages
        print(f"  → {len(poor_pages)}/{total_pages} pages need OCR ({poor_page_ratio:.1%})")
        
        # Use parallel processing if beneficial (2+ pages and enabled)
        # Lower threshold since CPU-bound processes benefit more from parallelization
        if use_parallel_ocr and len(poor_pages) >= 2:
            log.info("Using parallel OCR for {} pages...", len(poor_pages))
            ocr_results = extract_multiple_pages_ocr_parallel(
                pdf_path, poor_pages, dpi, lang, max_workers, ocr_method, gemini_config
            )
        else:
            # Sequential processing for small number of pages
            ocr_results = {}
            for page_num in poor_pages:
                try:
                    log.info("{} OCR processing page {}...", ocr_method.upper(), page_num + 1)
                    if ocr_method.lower() == "gemini" and GEMINI_AVAILABLE:
                        ocr_text = extract_single_page_ocr_gemini(pdf_path, page_num, gemini_config)
                    else:
                        ocr_text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
                    ocr_results[page_num] = ocr_text
                except Exception as e:
                    log.warning("{} OCR failed for page {}: {}", ocr_method.upper(), page_num + 1, e)
                    # Fallback for Gemini
                    if ocr_method.lower() == "gemini":
                        try:
                            log.info("Falling back to Tesseract for page {}", page_num + 1)
                            ocr_text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
                            ocr_results[page_num] = ocr_text
                        except Exception as e2:
                            log.warning("Tesseract fallback also failed for page {}: {}", page_num + 1, e2)
                            ocr_results[page_num] = ""
                    else:
                        ocr_results[page_num] = ""
        
        # Combine results: good PyMuPDF pages + OCR improved pages
        improved_pages = []
        ocr_count = 0
        
        for i, page_content in enumerate(pages):
            if i in poor_pages:
                ocr_text = ocr_results.get(i, "")
                if ocr_text.strip():
                    improved_pages.append(f"## Page {i+1}\n\n{ocr_text.strip()}\n")
                    ocr_count += 1
                else:
                    # OCR failed or empty, keep original
                    improved_pages.append(page_content)
                    if use_parallel_ocr and len(poor_pages) >= 2:
                        pass  # Already logged by parallel function
                    else:
                        log.info("OCR yielded no content for page {}, keeping original", i + 1)
            else:
                # Keep good PyMuPDF page
                improved_pages.append(page_content)
        
        # Combine improved pages
        hybrid_content = "\n\n".join(improved_pages)
        
        print(f"  → Hybrid extraction completed: {ocr_count}/{len(poor_pages)} pages improved with OCR")
        
        return hybrid_content, {
            "method": "hybrid_selective", 
            "total_pages": total_pages, 
            "ocr_pages": ocr_count,
            "poor_page_ratio": poor_page_ratio,
            "pages_attempted_ocr": len(poor_pages),
            "pages_improved_ocr": ocr_count,
            "parallel_processing": use_parallel_ocr and len(poor_pages) >= 2,
            "ocr_method": ocr_method,
            **base_stats
        }


def extract_text_with_ocr(pdf_path: str, dpi: int = 300, lang: str = 'eng') -> str:
    """Extract text from PDF using OCR via Tesseract with streaming processing.

    Uses PyMuPDF for rasterization when possible; on any PyMuPDF error, falls back to
    pdf2image per page. This avoids repeated MuPDF parser errors on malformed PDFs.
    """
    extracted_text: list[str] = []

    # Try PyMuPDF path first (unless forced to pdf2image)
    if _USE_FITZ_RENDERER:
        try:
            import fitz  # type: ignore
            doc = fitz.open(pdf_path)
            try:
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        pix.save(temp_file.name)
                        pix = None
                        try:
                            text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)
                            if text.strip():
                                extracted_text.append(f"## Page {page_num + 1}\n\n{text.strip()}\n")
                        finally:
                            os.unlink(temp_file.name)
                    if (page_num + 1) % 5 == 0:
                        import gc
                        gc.collect()
            finally:
                try:
                    doc.close()
                except Exception:
                    pass

            return "\n".join(extracted_text)
        except ImportError:
            # PyMuPDF not installed; fall through to pdf2image
            pass
        except Exception:
            # Any PyMuPDF runtime error → fallback to pdf2image path
            pass

    # Fallback: pdf2image sequential processing
    try:
        from pdf2image import convert_from_path
        # Determine page count by trying to convert first page, then iterate until conversion fails
        page_index = 1
        while True:
            pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_index, last_page=page_index)
            if not pages:
                break
            page_img = pages[0]
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                page_img.save(temp_file.name, 'PNG')
                page_img = None
                try:
                    text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)
                    if text.strip():
                        extracted_text.append(f"## Page {page_index}\n\n{text.strip()}\n")
                finally:
                    os.unlink(temp_file.name)
            if page_index % 5 == 0:
                import gc
                gc.collect()
            page_index += 1

        return "\n".join(extracted_text)
    except Exception as fallback_e:
        raise RuntimeError(f"OCR extraction failed for {pdf_path}: {fallback_e}")

def parse_pdf(pdf_path: str, output_format: str = "markdown", use_ocr: bool = False, 
              fallback_to_ocr: bool = True, ocr_lang: str = 'eng', 
              validation_config: ValidationConfig = None, use_hybrid: bool = True,
              max_workers: Optional[int] = None, use_parallel_ocr: bool = True,
              parser_method: str = "pymupdf", docling_config: DoclingConfig = None,
              ocr_method: str = "tesseract", gemini_config: GeminiConfig = None) -> str:
    """
    Parse PDF using intelligent hybrid approach: Primary parser + selective OCR for poor pages only.
    
    This implements the optimized workflow:
    1. Try primary parser first (PyMuPDF4LLM or Docling - fast, works for digital PDFs)
    2. Validate each page individually
    3. Apply OCR only to pages that failed validation
    4. Combine best results from both methods
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if validation_config is None:
        validation_config = ValidationConfig()
    
    try:
        # Force OCR if requested (skip hybrid processing)
        if use_ocr:
            print(f"  → Using full OCR extraction (forced)")
            return extract_text_with_ocr(pdf_path, lang=ocr_lang)
        
        if output_format != "markdown":
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Use hybrid approach by default
        if use_hybrid and fallback_to_ocr:
            content, stats = extract_text_hybrid(pdf_path, lang=ocr_lang, 
                                               validation_config=validation_config,
                                               max_workers=max_workers,
                                               use_parallel_ocr=use_parallel_ocr,
                                               parser_method=parser_method,
                                               docling_config=docling_config,
                                               ocr_method=ocr_method,
                                               gemini_config=gemini_config)
            
            # Log processing statistics
            method = stats.get('method', 'unknown')
            total_pages = stats.get('total_pages', 0)
            ocr_pages = stats.get('ocr_pages', 0)
            
            if method == 'pymupdf_only':
                print(f"  → Hybrid: All {total_pages} pages good quality, PyMuPDF only")
            elif method == 'full_ocr_threshold':
                print(f"  → Hybrid: {stats.get('poor_page_ratio', 0):.1%} pages poor, used full OCR")
            elif method == 'hybrid_selective':
                parallel_info = " (parallel)" if stats.get('parallel_processing') else " (sequential)"
                print(f"  → Hybrid: OCR applied to {ocr_pages}/{total_pages} pages ({ocr_pages/total_pages:.1%}){parallel_info}")
            else:
                parallel_info = " (parallel)" if stats.get('parallel_processing') else ""
                print(f"  → Hybrid extraction completed ({method}){parallel_info}")
            
            return content
        
        # Fallback to legacy approach if hybrid disabled
        print("  → Using legacy extraction approach...")
        content = pymupdf4llm.to_markdown(pdf_path)
        
        if fallback_to_ocr and content.strip():
            is_valid, validation_msg = validate_per_page_content(content, validation_config)
            print(f"  → Validation: {validation_msg}")
            
            if is_valid:
                print("  → PyMuPDF4LLM extraction passed validation")
                return content
            else:
                print("  → Validation failed, falling back to full OCR...")
                ocr_content = extract_text_with_ocr(pdf_path, lang=ocr_lang)
                if ocr_content.strip():
                    print("  → OCR extraction successful")
                    return ocr_content
                else:
                    print("  → OCR extraction yielded no content")
                    return content
        elif fallback_to_ocr:
            print("  → Empty content, falling back to OCR...")
            ocr_content = extract_text_with_ocr(pdf_path, lang=ocr_lang)
            return ocr_content if ocr_content.strip() else content
        
        return content
        
    except Exception as e:
        if fallback_to_ocr and not use_ocr:
            print(f"  → Primary extraction failed ({e}), attempting OCR fallback...")
            try:
                ocr_content = extract_text_with_ocr(pdf_path, lang=ocr_lang)
                print("  → OCR fallback successful")
                return ocr_content
            except Exception as ocr_e:
                raise RuntimeError(f"All extraction methods failed - Primary: {e}, OCR: {ocr_e}")
        raise RuntimeError(f"Failed to parse PDF {pdf_path}: {e}")

def find_pdf_files(documents_dir: str, recursive: bool = True, max_files: int = None) -> list:
    """Find PDF files in directory, optionally recursive with file limit."""
    documents_path = Path(documents_dir)
    
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(documents_path.glob(pattern))
    
    if max_files and len(pdf_files) > max_files:
        print(f"Found {len(pdf_files)} PDFs, limiting to {max_files} for processing")
        pdf_files = pdf_files[:max_files]
    
    return pdf_files

def process_documents_folder(documents_dir: str = "documents", output_format: str = "markdown", 
                           recursive: bool = True, max_files: int = None, use_ocr: bool = False,
                           fallback_to_ocr: bool = True, ocr_lang: str = 'eng',
                           validation_config: ValidationConfig = None, 
                           extract_metadata: bool = True, use_cache: bool = True,
                           cache_dir: str = ".cache/documents",
                           parser_method: str = "pymupdf", docling_config: DoclingConfig = None,
                           ocr_method: str = "tesseract", gemini_config: GeminiConfig = None) -> dict:
    """Process PDF files in the documents folder with enhanced metadata extraction and caching."""
    pdf_files = find_pdf_files(documents_dir, recursive, max_files)
    
    if not pdf_files:
        search_type = "recursively" if recursive else "in directory"
        print(f"No PDF files found {search_type} in {documents_dir}")
        return {}
    
    # Initialize cache if enabled
    document_cache = None
    if use_cache:
        from ..caching.document_cache import get_document_cache
        document_cache = get_document_cache(cache_dir)
        print(f"Cache enabled: {cache_dir}")
    
    results = {}
    cache_hits = 0
    cache_misses = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        relative_path = pdf_file.relative_to(Path(documents_dir))
        print(f"Processing ({i}/{len(pdf_files)}): {relative_path}")
        
        # Check cache first
        cached_result = None
        if document_cache:
            cached_result = document_cache.get_cached_result(str(pdf_file))
            
        if cached_result:
            # Use cached result
            content, metadata, processing_stats = cached_result
            cache_hits += 1
            
            if extract_metadata:
                results[str(relative_path)] = {
                    'content': content,
                    'metadata': metadata,
                    'processing_method': 'cached'
                }
            else:
                results[str(relative_path)] = content
                
        else:
            # Parse document
            cache_misses += 1
            try:
                import time
                start_time = time.time()
                
                content = parse_pdf(str(pdf_file), output_format, use_ocr, fallback_to_ocr, ocr_lang, validation_config,
                                   parser_method=parser_method, docling_config=docling_config,
                                   ocr_method=ocr_method, gemini_config=gemini_config)
                
                processing_time = time.time() - start_time
                processing_stats = {
                    'processing_time_seconds': processing_time,
                    'use_ocr': use_ocr,
                    'fallback_to_ocr': fallback_to_ocr,
                    'validation_config': asdict(validation_config) if validation_config else None
                }
                
                if extract_metadata:
                    # Import here to avoid circular imports
                    try:
                        from .enhanced_metadata import InformatikaMetadataExtractor
                        extractor = InformatikaMetadataExtractor()
                        metadata = extractor.extract_metadata(str(pdf_file), content[:1000] if content else "")
                        metadata_dict = metadata.to_dict()
                        
                        results[str(relative_path)] = {
                            'content': content,
                            'metadata': metadata_dict,
                            'processing_method': 'enhanced_with_metadata'
                        }
                        
                        # Cache the result
                        if document_cache and content:
                            document_cache.cache_result(str(pdf_file), content, metadata_dict, processing_stats)
                            
                    except ImportError:
                        metadata_dict = {'source_path': str(relative_path)}
                        results[str(relative_path)] = {
                            'content': content,
                            'metadata': metadata_dict,
                            'processing_method': 'basic'
                        }
                        
                        if document_cache and content:
                            document_cache.cache_result(str(pdf_file), content, metadata_dict, processing_stats)
                else:
                    results[str(relative_path)] = content
                    if document_cache and content:
                        basic_metadata = {'source_path': str(relative_path)}
                        document_cache.cache_result(str(pdf_file), content, basic_metadata, processing_stats)
                
                print(f"Successfully parsed {relative_path} ({processing_time:.1f}s)")
                
            except Exception as e:
                print(f"Error parsing {relative_path}: {e}")
                results[str(relative_path)] = None
    
    if document_cache:
        total_requests = cache_hits + cache_misses
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        print("\nCache Statistics:")
        print(f"   Cache hits: {cache_hits}/{total_requests} ({hit_rate:.1f}%)")
        print(f"   Time saved: ~{cache_hits * 15:.0f}s (estimated)")  # Rough estimate
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Parse PDF files using pymupdf4llm")
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Parse a specific PDF file"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="documents",
        help="Directory containing PDF files (default: documents)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown"],
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file to save results"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Search recursively in subdirectories (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive search"
    )
    parser.add_argument(
        "--max-docs", "-m",
        type=int,
        help="Maximum number of documents to process (for testing)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Force OCR extraction instead of standard text extraction"
    )
    parser.add_argument(
        "--no-ocr-fallback",
        action="store_true",
        help="Disable automatic OCR fallback when standard extraction fails"
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="eng",
        help="Language for OCR (default: eng, see tesseract docs for options)"
    )
    parser.add_argument(
        "--parser-method",
        type=str,
        default="pymupdf",
        choices=["pymupdf", "docling"],
        help="Primary parser method: pymupdf (PyMuPDF4llm) or docling (default: pymupdf)"
    )
    parser.add_argument(
        "--docling-tables",
        action="store_true",
        default=True,
        help="Enable table extraction in Docling (default: True)"
    )
    parser.add_argument(
        "--no-docling-tables",
        action="store_false",
        dest="docling_tables",
        help="Disable table extraction in Docling"
    )
    parser.add_argument(
        "--docling-figures",
        action="store_true", 
        default=True,
        help="Enable figure extraction in Docling (default: True)"
    )
    parser.add_argument(
        "--no-docling-figures",
        action="store_false",
        dest="docling_figures",
        help="Disable figure extraction in Docling"
    )
    parser.add_argument(
        "--docling-images",
        action="store_true",
        default=False,
        help="Enable image extraction in Docling (default: False)"
    )
    parser.add_argument(
        "--docling-table-mode",
        type=str,
        default="fast",
        choices=["fast", "accurate", "hybrid"],
        help="Table extraction mode for Docling (default: fast)"
    )
    parser.add_argument(
        "--ocr-method",
        type=str,
        default="tesseract",
        choices=["tesseract", "gemini"],
        help="OCR method: tesseract (local) or gemini (AI-powered, requires API key) (default: tesseract)"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        help="Gemini API key (or set GOOGLE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash-lite",
        help="Gemini model to use for OCR (default: gemini-2.0-flash-lite)"
    )
    parser.add_argument(
        "--gemini-retries",
        type=int,
        default=2,
        help="Maximum retries for Gemini API calls (default: 2)"
    )
    parser.add_argument(
        "--gemini-daily-limit",
        type=int,
        default=1500,
        help="Daily request limit for Gemini enforcement; 0 disables (default: 1500)"
    )
    parser.add_argument(
        "--disable-gemini-limit",
        action="store_true",
        help="Disable Gemini daily limit enforcement for this run"
    )
    parser.add_argument(
        "--gemini-usage-dir",
        type=str,
        default=".cache/api_usage",
        help="Directory for storing Gemini API usage stats (default: .cache/api_usage)"
    )
    parser.add_argument(
        "--gemini-estimated-cost-per-page",
        type=float,
        default=0.0,
        help="Optional cost estimate added per Gemini page request in USD (default: 0.0)"
    )
    parser.add_argument(
        "--print-gemini-usage",
        action="store_true",
        help="Print Gemini API usage summary at end of run"
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum characters per page for validation (default: 20)"
    )
    parser.add_argument(
        "--min-word-ratio",
        type=float,
        default=0.3,
        help="Minimum word ratio for validation (default: 0.3)"
    )
    parser.add_argument(
        "--max-symbol-ratio",
        type=float,
        default=0.3,
        help="Maximum symbol ratio for validation (default: 0.3)"
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.5,
        help="Minimum alphabetic character ratio for validation (default: 0.5)"
    )
    parser.add_argument(
        "--disable-hybrid",
        action="store_true",
        help="Disable hybrid extraction (use legacy all-or-nothing approach)"
    )
    parser.add_argument(
        "--disable-parallel",
        action="store_true",
        help="Disable parallel OCR processing (use sequential)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=f"Maximum number of parallel OCR workers (default: {get_physical_cpu_cores()} physical cores)"
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable document caching (always reprocess files)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/documents",
        help="Directory for document cache (default: .cache/documents)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear document cache and exit"
    )
    
    args = parser.parse_args()
    
    # Create validation configuration from CLI args
    validation_config = ValidationConfig(
        min_chars_per_page=args.min_chars,
        min_word_ratio=args.min_word_ratio,
        max_symbol_ratio=args.max_symbol_ratio,
        min_alpha_ratio=args.min_alpha_ratio
    )
    
    # Create Docling configuration from CLI args
    docling_config = DoclingConfig(
        extract_tables=args.docling_tables,
        extract_images=args.docling_images, 
        extract_figures=args.docling_figures,
        table_extraction_mode=args.docling_table_mode,
        ocr_enabled=False  # We handle OCR separately in hybrid mode
    )
    
    # Create Gemini configuration from CLI args
    daily_limit = 0 if args.disable_gemini_limit else args.gemini_daily_limit
    # Set usage directory env for the process so all worker processes pick it up
    try:
        os.environ["GEMINI_USAGE_DIR"] = args.gemini_usage_dir
    except Exception:
        pass

    gemini_config = GeminiConfig(
        model=args.gemini_model,
        api_key=args.gemini_api_key,
        max_retries=args.gemini_retries,
        timeout_seconds=30,
        temperature=0.0,
        daily_limit=daily_limit,
        usage_dir=args.gemini_usage_dir,
        estimated_cost_per_page_usd=args.gemini_estimated_cost_per_page
    )
    
    try:
        # Handle cache operations first
        if args.clear_cache:
            from ..caching.document_cache import get_document_cache
            cache = get_document_cache(args.cache_dir)
            cache.clear_cache()
            cache.print_cache_stats()
            print("Cache cleared successfully.")
            return 0
        
        if args.file:
            hybrid_status = "disabled" if args.disable_hybrid else "enabled"
            print(f"Parsing single file: {args.file} (hybrid: {hybrid_status})")
            content = parse_pdf(args.file, args.format, args.ocr, not args.no_ocr_fallback, 
                              args.ocr_lang, validation_config, use_hybrid=not args.disable_hybrid,
                              parser_method=args.parser_method, docling_config=docling_config,
                              ocr_method=args.ocr_method, gemini_config=gemini_config)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Results saved to: {args.output}")
            else:
                print("\n" + "="*50)
                print(content)
        else:
            search_mode = "recursively" if args.recursive else "non-recursively"
            max_info = f" (max {args.max_docs})" if args.max_docs else ""
            ocr_info = " (OCR mode)" if args.ocr else " (auto OCR fallback)" if not args.no_ocr_fallback else ""
            cache_info = " (cache disabled)" if args.disable_cache else f" (cache: {args.cache_dir})"
            print(f"Processing PDFs {search_mode} in: {args.dir}{max_info}{ocr_info}{cache_info}")
            results = process_documents_folder(
                args.dir, args.format, args.recursive, args.max_docs, 
                args.ocr, not args.no_ocr_fallback, args.ocr_lang, validation_config,
                extract_metadata=True, use_cache=not args.disable_cache, cache_dir=args.cache_dir,
                parser_method=args.parser_method, docling_config=docling_config,
                ocr_method=args.ocr_method, gemini_config=gemini_config
            )
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for filename, content in results.items():
                        if content:
                            f.write(f"\n# {filename}\n\n")
                            f.write(content)
                            f.write("\n\n" + "="*80 + "\n\n")
                print(f"Results saved to: {args.output}")
            else:
                for filename, content in results.items():
                    if content:
                        print(f"\n# {filename}")
                        print("="*50)
                        print(content[:500] + "..." if len(content) > 500 else content)
                        print("\n")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    # Optionally print usage summary
    try:
        if args.print_gemini_usage and get_gemini_usage_tracker is not None:
            tracker = get_gemini_usage_tracker()
            tracker.print_summary()
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    exit(main())