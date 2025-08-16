from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPyMuPDFConfig:
    """Configuration for enhanced PyMuPDF parsing with advanced features."""
    
    # Text extraction options
    extract_text_blocks: bool = True
    extract_text_words: bool = True
    extract_font_info: bool = True
    preserve_layout: bool = True
    extract_hyperlinks: bool = True
    
    # Table and structure detection
    extract_tables: bool = True
    table_strategy: str = "lines_strict"  # "lines_strict", "lines", "text"
    
    # Image and drawing extraction
    extract_images: bool = True
    extract_drawings: bool = True
    min_image_size: int = 50  # Minimum dimension to extract images
    
    # Advanced text analysis
    extract_reading_order: bool = True
    detect_columns: bool = True
    merge_text_blocks: bool = True
    
    # Performance options
    use_text_flags: bool = True
    clip_to_mediabox: bool = True
    
    # Output formatting
    output_format: str = "enhanced"  # "basic", "enhanced", "structured"


@dataclass
class DocumentStructure:
    """Enhanced document structure with detailed extraction results."""
    text_content: str
    text_blocks: List[Dict[str, Any]]
    text_words: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    drawings: List[Dict[str, Any]]
    hyperlinks: List[Dict[str, Any]]
    fonts: List[Dict[str, Any]]
    page_layouts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def extract_enhanced_text(page: "fitz.Page", config: EnhancedPyMuPDFConfig) -> Dict[str, Any]:
    """Extract text with enhanced formatting and structure preservation."""
    result = {
        "text": "",
        "blocks": [],
        "words": [],
        "fonts": [],
        "reading_order": []
    }
    
    # Use advanced text extraction flags
    text_flags = 0
    if config.use_text_flags:
        text_flags = (
            fitz.TEXTFLAGS_TEXT |
            fitz.TEXTFLAGS_BLOCKS |
            fitz.TEXTFLAGS_WORDS |
            fitz.TEXT_PRESERVE_WHITESPACE |
            fitz.TEXT_PRESERVE_LIGATURES
        )
        
        if config.preserve_layout:
            text_flags |= fitz.TEXT_PRESERVE_SPANS
            
    # Extract structured text data
    text_dict = page.get_text("dict", flags=text_flags)
    
    # Process text blocks with enhanced information
    if config.extract_text_blocks:
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                block_info = {
                    "bbox": block["bbox"],
                    "block_no": block["number"],
                    "type": "text",
                    "lines": []
                }
                
                for line in block["lines"]:
                    line_info = {
                        "bbox": line["bbox"],
                        "wmode": line.get("wmode", 0),  # Writing mode
                        "dir": line.get("dir", (1, 0)),  # Text direction
                        "spans": []
                    }
                    
                    for span in line["spans"]:
                        span_info = {
                            "bbox": span["bbox"],
                            "text": span["text"],
                            "font": span["font"],
                            "size": span["size"],
                            "flags": span["flags"],
                            "color": span.get("color", 0),
                            "ascender": span.get("ascender", 0),
                            "descender": span.get("descender", 0)
                        }
                        line_info["spans"].append(span_info)
                        
                        # Collect font information
                        if config.extract_font_info:
                            font_info = {
                                "name": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],
                                "bbox": span["bbox"]
                            }
                            result["fonts"].append(font_info)
                    
                    block_info["lines"].append(line_info)
                result["blocks"].append(block_info)
            
            elif "image" in block:  # Image block
                block_info = {
                    "bbox": block["bbox"],
                    "block_no": block["number"],
                    "type": "image",
                    "image": block["image"]
                }
                result["blocks"].append(block_info)
    
    # Extract word-level information
    if config.extract_text_words:
        words = page.get_text("words")
        for word in words:
            word_info = {
                "bbox": word[:4],
                "text": word[4],
                "block_no": word[5] if len(word) > 5 else 0,
                "line_no": word[6] if len(word) > 6 else 0,
                "word_no": word[7] if len(word) > 7 else 0
            }
            result["words"].append(word_info)
    
    # Extract basic text
    result["text"] = page.get_text()
    
    return result


def extract_tables_advanced(page: "fitz.Page", config: EnhancedPyMuPDFConfig) -> List[Dict[str, Any]]:
    """Extract tables using PyMuPDF's advanced table detection."""
    tables = []
    
    if not config.extract_tables:
        return tables
    
    try:
        # Use PyMuPDF's built-in table detection
        found_tables = page.find_tables(strategy=config.table_strategy)
        
        for table_idx, table in enumerate(found_tables):
            table_info = {
                "table_id": table_idx,
                "bbox": table.bbox,
                "cells": [],
                "header": None,
                "rows": table.extract(),
                "col_count": len(table.header.names) if table.header else 0,
                "row_count": len(table.extract())
            }
            
            # Extract header information
            if table.header:
                table_info["header"] = {
                    "names": table.header.names,
                    "bbox": table.header.bbox
                }
            
            # Extract cell-level information
            for row_idx, row in enumerate(table.extract()):
                for col_idx, cell_value in enumerate(row):
                    cell_info = {
                        "row": row_idx,
                        "col": col_idx,
                        "value": cell_value,
                        "bbox": table.cells[row_idx][col_idx] if row_idx < len(table.cells) and col_idx < len(table.cells[row_idx]) else None
                    }
                    table_info["cells"].append(cell_info)
            
            tables.append(table_info)
            
    except Exception as e:
        logger.warning(f"Table extraction failed: {e}")
    
    return tables


def extract_images_and_drawings(page: "fitz.Page", config: EnhancedPyMuPDFConfig) -> Tuple[List[Dict], List[Dict]]:
    """Extract images and vector drawings from the page."""
    images = []
    drawings = []
    
    # Extract images
    if config.extract_images:
        try:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                img_info = {
                    "index": img_index,
                    "xref": img[0],
                    "smask": img[1],
                    "width": img[2],
                    "height": img[3],
                    "bpc": img[4],  # Bits per component
                    "colorspace": img[5],
                    "alt": img[6],  # Alternate colorspace
                    "name": img[7],
                    "filter": img[8],
                    "bbox": None
                }
                
                # Skip small images if configured
                if (img_info["width"] < config.min_image_size or 
                    img_info["height"] < config.min_image_size):
                    continue
                
                # Get image bbox on page
                try:
                    img_rects = page.get_image_rects(img[0])
                    if img_rects:
                        img_info["bbox"] = img_rects[0]
                except:
                    pass
                
                images.append(img_info)
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
    
    # Extract vector drawings
    if config.extract_drawings:
        try:
            paths = page.get_drawings()
            for path_index, path in enumerate(paths):
                drawing_info = {
                    "index": path_index,
                    "bbox": path.get("rect"),
                    "items": path.get("items", []),
                    "type": "vector_drawing",
                    "fill": path.get("fill"),
                    "stroke": path.get("stroke"),
                    "width": path.get("width", 0)
                }
                drawings.append(drawing_info)
        except Exception as e:
            logger.warning(f"Drawing extraction failed: {e}")
    
    return images, drawings


def extract_hyperlinks(page: "fitz.Page", config: EnhancedPyMuPDFConfig) -> List[Dict[str, Any]]:
    """Extract hyperlinks and annotations from the page."""
    links = []
    
    if not config.extract_hyperlinks:
        return links
    
    try:
        # Extract links
        link_list = page.get_links()
        for link in link_list:
            link_info = {
                "kind": link.get("kind"),
                "from": link.get("from"),  # Source rectangle
                "to": link.get("to"),      # Destination
                "uri": link.get("uri"),    # URL for web links
                "page": link.get("page"),  # Target page for internal links
                "zoom": link.get("zoom"),  # Zoom factor
                "title": link.get("title", "")
            }
            links.append(link_info)
    
    except Exception as e:
        logger.warning(f"Link extraction failed: {e}")
    
    return links


def analyze_page_layout(page: "fitz.Page", config: EnhancedPyMuPDFConfig) -> Dict[str, Any]:
    """Analyze page layout and structure."""
    layout = {
        "page_number": page.number,
        "bbox": page.bound(),
        "rotation": page.rotation,
        "mediabox": page.mediabox,
        "cropbox": page.cropbox,
        "columns": [],
        "text_regions": [],
        "reading_order": []
    }
    
    if config.detect_columns:
        # Simple column detection based on text block positions
        text_blocks = page.get_text("dict")["blocks"]
        
        # Group blocks by approximate x-position for column detection
        text_positions = []
        for block in text_blocks:
            if "lines" in block:
                bbox = block["bbox"]
                text_positions.append({
                    "left": bbox[0],
                    "right": bbox[2],
                    "top": bbox[1],
                    "bottom": bbox[3],
                    "center_x": (bbox[0] + bbox[2]) / 2
                })
        
        # Simple clustering by x-position for column detection
        if text_positions:
            text_positions.sort(key=lambda x: x["center_x"])
            
            # Basic column boundary detection
            page_width = page.bound().width
            if len(text_positions) > 1:
                # Check if there are clear column boundaries
                x_positions = [pos["center_x"] for pos in text_positions]
                gaps = []
                
                for i in range(1, len(x_positions)):
                    gap = x_positions[i] - x_positions[i-1]
                    if gap > page_width * 0.1:  # 10% of page width
                        gaps.append(x_positions[i-1] + gap/2)
                
                if gaps:
                    layout["columns"] = gaps
    
    return layout


def extract_with_enhanced_pymupdf(
    pdf_path: str, 
    config: EnhancedPyMuPDFConfig = None
) -> DocumentStructure:
    """Extract document content using enhanced PyMuPDF features."""
    
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF not available")
    
    if config is None:
        config = EnhancedPyMuPDFConfig()
    
    doc = fitz.open(pdf_path)
    
    # Initialize result structure
    all_text = []
    all_text_blocks = []
    all_text_words = []
    all_tables = []
    all_images = []
    all_drawings = []
    all_hyperlinks = []
    all_fonts = []
    all_page_layouts = []
    
    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # Extract enhanced text
            text_result = extract_enhanced_text(page, config)
            all_text.append(text_result["text"])
            all_text_blocks.extend(text_result["blocks"])
            all_text_words.extend(text_result["words"])
            all_fonts.extend(text_result["fonts"])
            
            # Extract tables
            if config.extract_tables:
                page_tables = extract_tables_advanced(page, config)
                all_tables.extend(page_tables)
            
            # Extract images and drawings
            if config.extract_images or config.extract_drawings:
                images, drawings = extract_images_and_drawings(page, config)
                all_images.extend(images)
                all_drawings.extend(drawings)
            
            # Extract hyperlinks
            if config.extract_hyperlinks:
                links = extract_hyperlinks(page, config)
                all_hyperlinks.extend(links)
            
            # Analyze page layout
            layout = analyze_page_layout(page, config)
            all_page_layouts.append(layout)
    
    finally:
        doc.close()
    
    # Extract document metadata
    doc = fitz.open(pdf_path)
    try:
        metadata = doc.metadata
        metadata.update({
            "page_count": doc.page_count,
            "is_pdf": doc.is_pdf,
            "needs_pass": doc.needs_pass,
            "permissions": doc.permissions if hasattr(doc, 'permissions') else None,
            "is_form_pdf": doc.is_form_pdf if hasattr(doc, 'is_form_pdf') else False
        })
    finally:
        doc.close()
    
    return DocumentStructure(
        text_content="\n\n".join(all_text),
        text_blocks=all_text_blocks,
        text_words=all_text_words,
        tables=all_tables,
        images=all_images,
        drawings=all_drawings,
        hyperlinks=all_hyperlinks,
        fonts=all_fonts,
        page_layouts=all_page_layouts,
        metadata=metadata
    )


def format_enhanced_output(
    structure: DocumentStructure, 
    format_type: str = "enhanced"
) -> str:
    """Format the extracted structure into enhanced markdown output."""
    
    if format_type == "basic":
        return structure.text_content
    
    output = []
    
    # Add metadata header
    if structure.metadata:
        output.append("# Document Information")
        output.append(f"- **Pages**: {structure.metadata.get('page_count', 'Unknown')}")
        output.append(f"- **Title**: {structure.metadata.get('title', 'Untitled')}")
        output.append(f"- **Author**: {structure.metadata.get('author', 'Unknown')}")
        output.append(f"- **Subject**: {structure.metadata.get('subject', 'N/A')}")
        output.append("")
    
    # Add main text content
    output.append("# Document Content")
    output.append(structure.text_content)
    output.append("")
    
    # Add tables if found
    if structure.tables:
        output.append("# Tables")
        for i, table in enumerate(structure.tables):
            output.append(f"## Table {i+1}")
            if table["header"]:
                headers = table["header"]["names"]
                output.append("| " + " | ".join(headers) + " |")
                output.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            for row in table["rows"]:
                output.append("| " + " | ".join(str(cell) for cell in row) + " |")
            output.append("")
    
    # Add images information
    if structure.images:
        output.append("# Images")
        for i, img in enumerate(structure.images):
            output.append(f"- **Image {i+1}**: {img['width']}x{img['height']} pixels")
            if img.get("name"):
                output.append(f"  - Name: {img['name']}")
        output.append("")
    
    # Add hyperlinks
    if structure.hyperlinks:
        output.append("# Links")
        for link in structure.hyperlinks:
            if link.get("uri"):
                output.append(f"- [{link.get('title', 'Link')}]({link['uri']})")
            elif link.get("page"):
                output.append(f"- Internal link to page {link['page']}")
        output.append("")
    
    return "\n".join(output)


# Example usage functions
def create_enhanced_pymupdf_config(**kwargs) -> EnhancedPyMuPDFConfig:
    """Create configuration with custom settings."""
    return EnhancedPyMuPDFConfig(**kwargs)


def extract_pdf_enhanced(pdf_path: str, **config_kwargs) -> str:
    """Simple interface for enhanced PDF extraction."""
    config = create_enhanced_pymupdf_config(**config_kwargs)
    structure = extract_with_enhanced_pymupdf(pdf_path, config)
    return format_enhanced_output(structure, config.output_format)