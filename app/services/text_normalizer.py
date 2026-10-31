"""Text normalization service that preserves document structure and page demarcations."""

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class NormalizedDocument:
    """Document text normalized for LLM prompt ingestion and evidence alignment."""

    full_text: str
    pages: dict[int, str]  # 1-indexed page number -> normalized page text


class TextNormalizer:
    """Normalizes raw and PDF-extracted text while preserving useful document structure."""

    @staticmethod
    def normalize_string(text: str) -> str:
        """Cleans and standardizes raw text strings."""
        if not text:
            return ""

        # Normalize unicode characters (NFKC handles compatibility chars and ligatures)
        normalized = unicodedata.normalize("NFKC", text)

        # Standardize quotes, apostrophes, and hyphens/dashes
        replacements = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u2014": " - ",
            "\u2013": "-",
            "\u2212": "-",
            "\u00a0": " ",  # Non-breaking space
            "\t": "    ",
        }
        for orig, repl in replacements.items():
            normalized = normalized.replace(orig, repl)

        # Remove unprintable / control characters except newline
        normalized = "".join(
            ch for ch in normalized if ch == "\n" or not unicodedata.category(ch).startswith("C")
        )

        # Normalize line endings
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing whitespace on each line while preserving indentation/line breaks
        lines = [line.rstrip() for line in normalized.split("\n")]
        cleaned = "\n".join(lines)

        # Collapse excessive blank lines (more than 2 consecutive newlines) into 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def normalize_pages(self, pages: list[tuple[int, str]]) -> NormalizedDocument:
        """Combines multiple pages into a cohesive document with clear page demarcations.

        Args:
            pages: List of (page_number, raw_text) tuples where page_number is 1-indexed.

        Returns:
            NormalizedDocument with combined full_text and page lookup.
        """
        page_dict: dict[int, str] = {}
        combined_segments: list[str] = []

        for page_num, raw_page_text in pages:
            clean_page = self.normalize_string(raw_page_text)
            page_dict[page_num] = clean_page

            if clean_page:
                combined_segments.append(f"--- [Page {page_num}] ---\n{clean_page}")

        full_text = "\n\n".join(combined_segments).strip()
        return NormalizedDocument(full_text=full_text, pages=page_dict)

    def normalize_raw_text(self, text: str) -> NormalizedDocument:
        """Normalizes a raw text string, detecting any embedded page markers if present."""
        clean_text = self.normalize_string(text)

        # Check for user-provided page demarcations like '--- Page 1 ---' or '[Page 1]'
        page_pattern = re.compile(
            r"(?:^|\n)(?:[-=]{2,}\s*)?\[?Page\s+(\d+)\]?(?:\s*[-=]{2,})?", re.IGNORECASE
        )
        splits = page_pattern.split(clean_text)

        pages: dict[int, str] = {}
        if len(splits) > 1:
            # We have demarcations: [preamble, page_num1, text1, page_num2, text2, ...]
            preamble = splits[0].strip()
            if preamble:
                pages[1] = preamble

            for i in range(1, len(splits), 2):
                p_num = int(splits[i])
                p_text = splits[i + 1].strip() if i + 1 < len(splits) else ""
                pages[p_num] = p_text
        else:
            pages[1] = clean_text

        # Re-build normalized full text with consistent page tags
        return self.normalize_pages(list(pages.items()))
