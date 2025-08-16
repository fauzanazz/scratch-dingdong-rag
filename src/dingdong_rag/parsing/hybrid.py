from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from .validation import (
    ValidationConfig,
    ValidationResult,
    validate_extracted_text,
    validate_per_page_content,
    split_text_by_pages,
)
from .docling_adapter import DoclingConfig, extract_text_with_docling, DOCLING_AVAILABLE
from .ocr import (
    GeminiConfig,
    extract_multiple_pages_ocr_parallel,
    extract_single_page_ocr,
    extract_single_page_ocr_gemini,
)
import pymupdf4llm


@dataclass
class HybridOptions:
    parser_method: str = "pymupdf"  # or "docling"
    ocr_method: str = "tesseract"  # or "gemini"
    use_parallel_ocr: bool = True
    max_workers: Optional[int] = None


def extract_text_hybrid(
    pdf_path: str,
    dpi: int = 300,
    lang: str = "eng",
    validation_config: ValidationConfig | None = None,
    max_workers: Optional[int] = None,
    use_parallel_ocr: bool = True,
    parser_method: str = "pymupdf",
    docling_config: DoclingConfig | None = None,
    ocr_method: str = "tesseract",
    gemini_config: GeminiConfig | None = None,
) -> Tuple[str, Dict[str, any]]:
    if validation_config is None:
        validation_config = ValidationConfig()

    # 1) Primary extraction
    content, base_stats = _primary_extract(pdf_path, parser_method, docling_config, dpi, lang)

    if not content.strip():
        return _full_ocr(pdf_path, dpi, lang, ocr_method, gemini_config), {
            "method": "full_ocr_empty",
            "ocr_pages": "all",
            **base_stats,
        }

    # 2) Validate pages
    pages = split_text_by_pages(content)
    total_pages = len(pages)

    if total_pages <= 1:
        validation = validate_extracted_text(content, validation_config)
        if validation.is_valid:
            return content, {"method": "single_page_success", "total_pages": 1, "ocr_pages": 0, **base_stats}
        ocr_content = _full_ocr(pdf_path, dpi, lang, ocr_method, gemini_config)
        return ocr_content, {"method": "single_page_ocr", "total_pages": 1, "ocr_pages": 1, **base_stats}

    poor_pages = [i for i, p in enumerate(pages) if not validate_extracted_text(p, validation_config).is_valid]
    poor_ratio = len(poor_pages) / total_pages

    if not poor_pages:
        return content, {"method": "primary_parser_only", "total_pages": total_pages, "ocr_pages": 0, "poor_page_ratio": 0.0, **base_stats}

    if poor_ratio > 0.7:
        ocr_content = _full_ocr(pdf_path, dpi, lang, ocr_method, gemini_config, use_parallel_ocr, max_workers)
        return ocr_content, {
            "method": "full_ocr_threshold",
            "total_pages": total_pages,
            "ocr_pages": total_pages,
            "poor_page_ratio": poor_ratio,
            "parallel_processing": use_parallel_ocr,
            "ocr_method": ocr_method,
            **base_stats,
        }

    # selective OCR for poor pages
    if use_parallel_ocr and len(poor_pages) >= 2:
        ocr_results = extract_multiple_pages_ocr_parallel(
            pdf_path, poor_pages, dpi, lang, max_workers, ocr_method, gemini_config
        )
    else:
        ocr_results = {}
        for page_num in poor_pages:
            if ocr_method.lower() == "gemini":
                ocr_text = extract_single_page_ocr_gemini(pdf_path, page_num, gemini_config)
            else:
                ocr_text = extract_single_page_ocr(pdf_path, page_num, dpi, lang)
            ocr_results[page_num] = ocr_text

    improved_pages: List[str] = []
    ocr_count = 0
    for i, page_content in enumerate(pages):
        if i in poor_pages:
            ocr_text = ocr_results.get(i, "")
            if ocr_text.strip():
                improved_pages.append(f"## Page {i+1}\n\n{ocr_text.strip()}\n")
                ocr_count += 1
            else:
                improved_pages.append(page_content)
        else:
            improved_pages.append(page_content)

    hybrid_content = "\n\n".join(improved_pages)
    return hybrid_content, {
        "method": "hybrid_selective",
        "total_pages": total_pages,
        "ocr_pages": ocr_count,
        "poor_page_ratio": poor_ratio,
        "pages_attempted_ocr": len(poor_pages),
        "pages_improved_ocr": ocr_count,
        "parallel_processing": use_parallel_ocr and len(poor_pages) >= 2,
        "ocr_method": ocr_method,
        **base_stats,
    }


def _primary_extract(
    pdf_path: str,
    parser_method: str,
    docling_config: DoclingConfig | None,
    dpi: int,
    lang: str,
) -> Tuple[str, Dict[str, any]]:
    stats: Dict[str, any] = {}
    if parser_method.lower() == "docling" and DOCLING_AVAILABLE:
        try:
            content, docling_stats = extract_text_with_docling(pdf_path, docling_config)
            stats.update(docling_stats)
            stats["primary_parser"] = "docling"
            return content, stats
        except Exception as exc:
            try:
                content = pymupdf4llm.to_markdown(pdf_path)
                stats["primary_parser"] = "pymupdf_fallback"
                return content, stats
            except Exception:
                return "", {"primary_parser": "failed"}
    elif parser_method.lower() == "docling" and not DOCLING_AVAILABLE:
        try:
            content = pymupdf4llm.to_markdown(pdf_path)
            stats["primary_parser"] = "pymupdf_fallback"
            return content, stats
        except Exception:
            return "", {"primary_parser": "failed"}
    else:
        try:
            content = pymupdf4llm.to_markdown(pdf_path)
            stats["primary_parser"] = "pymupdf"
            return content, stats
        except Exception:
            return "", {"primary_parser": "failed"}


def _full_ocr(
    pdf_path: str,
    dpi: int,
    lang: str,
    ocr_method: str,
    gemini_config: GeminiConfig | None,
    use_parallel_ocr: bool = True,
    max_workers: Optional[int] = None,
) -> str:
    if ocr_method.lower() == "gemini":
        try:
            import fitz

            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            pages = []
            for page_num in range(total_pages):
                page_text = extract_single_page_ocr_gemini(pdf_path, page_num, gemini_config)
                if page_text.strip():
                    pages.append(f"## Page {page_num + 1}\n\n{page_text.strip()}\n")
            return "\n".join(pages)
        except Exception:
            from .ocr import extract_text_with_ocr_fallback
            return extract_text_with_ocr_fallback(pdf_path, dpi, lang)

    if use_parallel_ocr:
        try:
            import fitz

            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
            page_numbers = list(range(total_pages))
            results = extract_multiple_pages_ocr_parallel(
                pdf_path, page_numbers, dpi, lang, max_workers, ocr_method, gemini_config
            )
            combined = []
            for i in range(total_pages):
                t = results.get(i, "")
                if t.strip():
                    combined.append(f"## Page {i + 1}\n\n{t.strip()}\n")
            return "\n".join(combined)
        except Exception:
            from .ocr import extract_text_with_ocr_fallback
            return extract_text_with_ocr_fallback(pdf_path, dpi, lang)

    from .ocr import extract_text_with_ocr_fallback

    return extract_text_with_ocr_fallback(pdf_path, dpi, lang)
