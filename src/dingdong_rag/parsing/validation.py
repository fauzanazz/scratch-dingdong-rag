from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ValidationConfig:
    min_chars_per_page: int = 20
    min_word_ratio: float = 0.3
    max_symbol_ratio: float = 0.3
    min_alpha_ratio: float = 0.5


@dataclass
class ValidationResult:
    is_valid: bool
    char_count: int
    word_ratio: float
    symbol_ratio: float
    alpha_ratio: float
    reason: str


def validate_extracted_text(text: str, config: ValidationConfig | None = None) -> ValidationResult:
    if config is None:
        config = ValidationConfig()

    if not text or not text.strip():
        return ValidationResult(False, 0, 0.0, 0.0, 0.0, "Empty or whitespace-only text")

    clean_text = text.strip()
    char_count = len(clean_text)

    if char_count < config.min_chars_per_page:
        return ValidationResult(
            False, char_count, 0.0, 0.0, 0.0,
            f"Insufficient characters ({char_count} < {config.min_chars_per_page})",
        )

    tokens = re.findall(r"\S+", clean_text)
    if not tokens:
        return ValidationResult(False, char_count, 0.0, 0.0, 0.0, "No tokens found")

    alpha_chars = sum(1 for c in clean_text if c.isalpha())
    digit_chars = sum(1 for c in clean_text if c.isdigit())
    space_chars = sum(1 for c in clean_text if c.isspace())
    symbol_chars = char_count - alpha_chars - digit_chars - space_chars

    alpha_ratio = alpha_chars / char_count if char_count > 0 else 0.0
    symbol_ratio = symbol_chars / char_count if char_count > 0 else 0.0

    word_count = sum(1 for token in tokens if re.match(r"^[a-zA-Z]{2,}", token))
    word_ratio = word_count / len(tokens) if tokens else 0.0

    if alpha_ratio < config.min_alpha_ratio:
        return ValidationResult(
            False, char_count, word_ratio, symbol_ratio, alpha_ratio,
            f"Too few alphabetic characters ({alpha_ratio:.2f} < {config.min_alpha_ratio})",
        )

    if symbol_ratio > config.max_symbol_ratio:
        return ValidationResult(
            False, char_count, word_ratio, symbol_ratio, alpha_ratio,
            f"Too many symbols/special characters ({symbol_ratio:.2f} > {config.max_symbol_ratio})",
        )

    if word_ratio < config.min_word_ratio:
        return ValidationResult(
            False, char_count, word_ratio, symbol_ratio, alpha_ratio,
            f"Too few recognizable words ({word_ratio:.2f} < {config.min_word_ratio})",
        )

    return ValidationResult(True, char_count, word_ratio, symbol_ratio, alpha_ratio, "Text quality acceptable")


def validate_per_page_content(content: str, config: ValidationConfig | None = None) -> Tuple[bool, str]:
    if config is None:
        config = ValidationConfig()

    page_patterns = [
        r"\n\s*Page \d+\s*\n",
        r"\n\s*## Page \d+\s*\n",
        r"\n\s*\d+\s*\n\s*\n",
        r"\f",
    ]

    pages: List[str] = [content]
    for pattern in page_patterns:
        if re.search(pattern, content):
            pages = re.split(pattern, content)
            break

    failed_pages = 0
    total_pages = len(pages)
    reasons: List[str] = []

    for i, page_content in enumerate(pages):
        if not page_content.strip():
            continue
        result = validate_extracted_text(page_content, config)
        if not result.is_valid:
            failed_pages += 1
            reasons.append(f"Page {i+1}: {result.reason}")

    failure_threshold = 0.3
    failure_rate = failed_pages / total_pages if total_pages > 0 else 1

    if failure_rate > failure_threshold:
        summary = f"Failed validation on {failed_pages}/{total_pages} pages ({failure_rate:.1%})"
        return False, f"{summary}. Issues: {'; '.join(reasons[:3])}"

    return True, f"Passed validation ({total_pages - failed_pages}/{total_pages} pages valid)"


def split_text_by_pages(content: str) -> List[str]:
    if not content.strip():
        return []

    page_patterns = [
        r"\n\s*## Page \d+\s*\n",
        r"\n\s*Page \d+\s*\n",
        r"\n\s*\d+\s*\n\s*\n",
        r"\f",
    ]

    for pattern in page_patterns:
        if re.search(pattern, content):
            pages = re.split(pattern, content)
            return [p.strip() for p in pages if p.strip()]

    return [content.strip()]
