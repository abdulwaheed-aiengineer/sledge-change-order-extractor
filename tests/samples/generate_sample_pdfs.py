"""Utility script to generate sample digital change-order PDF files for manual testing and curl demos."""

from pathlib import Path

import pymupdf

SAMPLES_DIR = Path(__file__).parent


def generate_pdf(text_filename: str, pdf_filename: str) -> None:
    """Generates a PDF document from a text sample file."""
    text_path = SAMPLES_DIR / text_filename
    if not text_path.exists():
        return

    content = text_path.read_text(encoding="utf-8")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # US Letter

    # Draw header line
    page.draw_rect(pymupdf.Rect(40, 40, 572, 45), fill=(0.1, 0.2, 0.4))

    # Insert text content
    page.insert_text(
        (50, 75),
        content,
        fontsize=9,
        fontname="helv",
    )

    pdf_out = SAMPLES_DIR / pdf_filename
    doc.save(str(pdf_out))
    doc.close()
    print(f"Generated {pdf_out}")


if __name__ == "__main__":
    generate_pdf("change_order_1_standard.txt", "change_order_1_standard.pdf")
    generate_pdf("change_order_2_deduction.txt", "change_order_2_deduction.pdf")
    generate_pdf("change_order_3_conflict.txt", "change_order_3_conflict.pdf")
    generate_pdf("change_order_4_missing_fields.txt", "change_order_4_missing_fields.pdf")
