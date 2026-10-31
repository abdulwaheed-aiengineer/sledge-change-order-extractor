"""Tests for text normalization and structure preservation."""

from app.services.text_normalizer import TextNormalizer


def test_normalize_unicode_and_special_characters() -> None:
    """Verifies standardizing smart quotes, em dashes, non-breaking spaces, and unicode."""
    normalizer = TextNormalizer()
    raw = "Project: \u201cOakridge\u201d \u2014 Cost: $10,000\u00a0USD"
    normalized = normalizer.normalize_string(raw)

    assert '"Oakridge"' in normalized
    assert " - " in normalized
    assert " $10,000 USD" in normalized or "$10,000 USD" in normalized


def test_preserves_document_layout_and_paragraphs() -> None:
    """Verifies that line breaks and table-like indentation are preserved while collapsing 3+ blank lines."""
    normalizer = TextNormalizer()
    text = (
        "Item 1: Electrical feeds\n"
        "Cost: $5,000.00\n\n\n\n\n"
        "Item 2: Plumbing reroute\n"
        "Cost: $2,000.00"
    )
    normalized = normalizer.normalize_string(text)

    # Excessive blank lines collapsed to 2
    assert "\n\n\n" not in normalized
    assert "Item 1: Electrical feeds\nCost: $5,000.00\n\nItem 2: Plumbing reroute" in normalized


def test_normalize_pages_adds_demarcations() -> None:
    """Verifies that multi-page inputs receive clear page boundaries."""
    normalizer = TextNormalizer()
    pages = [
        (1, "First page text regarding Oakridge."),
        (2, "Second page text regarding financial terms."),
    ]
    doc = normalizer.normalize_pages(pages)

    assert "--- [Page 1] ---" in doc.full_text
    assert "First page text regarding Oakridge." in doc.full_text
    assert "--- [Page 2] ---" in doc.full_text
    assert "Second page text regarding financial terms." in doc.full_text
    assert len(doc.pages) == 2


def test_normalize_raw_text_detects_page_markers() -> None:
    """Verifies that embedded page markers in raw text are parsed into separate pages."""
    normalizer = TextNormalizer()
    raw = "--- Page 1 ---\nProject: Harbor Point\n\n--- Page 2 ---\nCost Impact: $12,000.00"
    doc = normalizer.normalize_raw_text(raw)

    assert 1 in doc.pages
    assert 2 in doc.pages
    assert "Harbor Point" in doc.pages[1]
    assert "Cost Impact: $12,000.00" in doc.pages[2]
