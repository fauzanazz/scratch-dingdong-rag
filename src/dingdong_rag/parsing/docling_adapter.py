from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

def _suppress_torch_warnings():
    """Suppress PyTorch warnings that are not relevant for CPU-only processing."""
    warnings.filterwarnings('ignore', message=r".*pin_memory.*no accelerator is found.*")
    warnings.filterwarnings('ignore', message=r".*CUDA.*not available.*")
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    os.environ.setdefault('TORCH_HOME', os.path.join(os.getcwd(), '.cache', 'torch'))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # Force CPU mode

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
    
    try:
        from docling.document_converter import ConversionConfig
        _has_conversion_config = True
    except ImportError:
        _has_conversion_config = False
    
    try:
        from docling.datamodel.pipeline_options import TableFormerMode
        _has_table_former_mode = True
    except ImportError:
        _has_table_former_mode = False
    
    try:
        from docling.datamodel.base_models import InputFormat
        _has_input_format = True
    except ImportError:
        _has_input_format = False
        
except ImportError as e:
    DOCLING_AVAILABLE = False
    _docling_import_error = e
    _has_conversion_config = False
    _has_table_former_mode = False
    _has_input_format = False
except Exception as e:
    DOCLING_AVAILABLE = False
    _docling_import_error = e
    _has_conversion_config = False
    _has_table_former_mode = False
    _has_input_format = False


@dataclass
class DoclingConfig:
    extract_tables: bool = True
    extract_images: bool = False
    extract_figures: bool = True
    table_extraction_mode: str = "fast"  # "fast", "accurate", "hybrid"
    ocr_enabled: bool = False
    offline_mode: bool = False  # Skip model downloads, use only local models
    download_timeout: int = 300  # Timeout for model downloads in seconds
    max_download_retries: int = 3  # Max retries for failed downloads
    cache_dir: Optional[str] = None  # Custom cache directory for models
    suppress_warnings: bool = True  # Suppress PyTorch CPU warnings
    force_cpu: bool = True  # Force CPU mode (disable GPU detection)


def _configure_model_downloads(config: DoclingConfig) -> None:
    """Configure Hugging Face model download settings and PyTorch behavior."""
    # Configure cache directories
    if config.cache_dir:
        os.environ["HF_HOME"] = str(config.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(config.cache_dir)
    
    # Set download timeout
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(config.download_timeout)
    
    # Configure PyTorch behavior for CPU-only environments
    if config.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
        os.environ["OMP_NUM_THREADS"] = "1"  # Optimize for single-threaded CPU
    
    # Suppress warnings if requested
    if config.suppress_warnings:
        _suppress_torch_warnings()


def _create_docling_config(config: DoclingConfig) -> Optional[object]:
    """Create Docling ConversionConfig with appropriate settings."""
    if not DOCLING_AVAILABLE or not _has_conversion_config:
        return None
    
    try:
        # Create pipeline options dictionary
        pipeline_options = {}
        
        # Configure table extraction if TableFormerMode is available
        if _has_table_former_mode and config.extract_tables:
            try:
                # Map table extraction mode to Docling enum
                table_mode_map = {
                    "fast": TableFormerMode.FAST,
                    "accurate": TableFormerMode.ACCURATE, 
                    "hybrid": TableFormerMode.ACCURATE  # Use accurate for hybrid
                }
                
                table_mode = table_mode_map.get(config.table_extraction_mode, TableFormerMode.FAST)
                pipeline_options["do_table_structure"] = True
                pipeline_options["table_structure_options"] = {"mode": table_mode}
            except Exception as e:
                print(f"  Warning: Could not configure table extraction: {e}")
                pipeline_options["do_table_structure"] = config.extract_tables
        else:
            # Basic table extraction without advanced modes
            pipeline_options["do_table_structure"] = config.extract_tables
        
        # Configure figure extraction
        pipeline_options["do_figure"] = config.extract_figures
        
        # Configure OCR
        pipeline_options["do_ocr"] = config.ocr_enabled
        
        # Create conversion config
        conversion_config = ConversionConfig(
            pipeline_options=pipeline_options
        )
        
        return conversion_config
        
    except Exception as e:
        print(f"  Warning: Could not create advanced Docling config, using defaults: {e}")
        return None


def _retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
            print(f"  → Retrying in {delay:.1f}s...")
            time.sleep(delay)


def pre_download_models(config: DoclingConfig | None = None) -> bool:
    """Pre-download required models for Docling."""
    if not DOCLING_AVAILABLE:
        print("Docling not available for model pre-download")
        return False
        
    if config is None:
        config = DoclingConfig()
        
    print("Pre-downloading Docling models...")
    _configure_model_downloads(config)
    
    try:
        def download_models():
            converter = DocumentConverter()
            print("✓ Docling models downloaded successfully")
            return True
            
        return _retry_with_backoff(
            download_models, 
            max_retries=config.max_download_retries,
            base_delay=2.0
        )
        
    except Exception as e:
        print(f"Failed to pre-download models: {e}")
        return False


def extract_text_with_docling(pdf_path: str, config: DoclingConfig | None = None) -> Tuple[str, Dict[str, any]]:
    """Extract text from PDF using Docling with enhanced error handling and model download management."""
    if not DOCLING_AVAILABLE:
        raise ImportError(f"Docling is not available. Install with: uv add docling\nOriginal error: {_docling_import_error}")

    if config is None:
        config = DoclingConfig()

    _configure_model_downloads(config)
    
    print(f"  → Initializing Docling converter (offline_mode: {config.offline_mode})...")

    try:
        def create_converter():
            if config.offline_mode:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            docling_config = _create_docling_config(config)
            
            if docling_config:
                converter = DocumentConverter(config=docling_config)
            else:
                converter = DocumentConverter()
            
            return converter
        
        converter = _retry_with_backoff(
            create_converter,
            max_retries=config.max_download_retries,
            base_delay=1.0
        )
        
        print(f"  → Converting PDF: {Path(pdf_path).name}")
        
        source = Path(pdf_path)
        result = converter.convert(source)

        content = ""
        extraction_method = "unknown"
        
        if hasattr(result.document, "export_to_markdown"):
            content = result.document.export_to_markdown()
            extraction_method = "markdown"
        elif hasattr(result.document, "export_to_text"):
            content = result.document.export_to_text()
            extraction_method = "text"
        elif hasattr(result, "document") and hasattr(result.document, "main_text"):
            content = result.document.main_text
            extraction_method = "main_text"
        elif hasattr(result, "text"):
            content = result.text
            extraction_method = "text_attr"
        else:
            content = str(result.document) if hasattr(result, "document") else str(result)
            extraction_method = "string_fallback"

        stats: Dict[str, any] = {
            "method": "docling",
            "extraction_method": extraction_method,
            "tables_extracted": config.extract_tables,
            "figures_extracted": config.extract_figures,
            "images_extracted": config.extract_images,
            "table_mode": config.table_extraction_mode,
            "offline_mode": config.offline_mode,
        }

        if hasattr(result, "document") and hasattr(result.document, "pages"):
            stats["total_pages"] = len(result.document.pages)
        elif hasattr(result, "pages"):
            stats["total_pages"] = len(result.pages)

        print(f"Docling extraction completed ({extraction_method})")
        return content, stats
        
    except Exception as exc:
        error_msg = f"Docling extraction failed: {exc}"
        
        if "timeout" in str(exc).lower() or "read timed out" in str(exc).lower():
            error_msg += f"\nTry: --docling-offline-mode or increase --docling-download-timeout from {config.download_timeout}s"
        elif "connection" in str(exc).lower():
            error_msg += f"\nTry: --docling-offline-mode or check internet connection"
        elif "not found" in str(exc).lower() and "model" in str(exc).lower():
            error_msg += f"\nRun model pre-download first or use --docling-offline-mode"
            
        raise RuntimeError(error_msg)
