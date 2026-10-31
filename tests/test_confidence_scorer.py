"""Tests for heuristic confidence scoring engine."""

from app.models.change_order import RawFieldCandidate
from app.services.confidence_scorer import ConfidenceScorer


def test_missing_fields_receive_zero_confidence() -> None:
    """Verifies that missing fields strictly receive 0.0 confidence."""
    scorer = ConfidenceScorer()
    candidate = RawFieldCandidate(value=None, evidence=None)

    confidence = scorer.compute_field_confidence(
        field_name="contractor_name",
        value=None,
        raw_candidate=candidate,
        document_text="Sample text without contractor.",
    )

    assert confidence == 0.0


def test_direct_vs_inferred_baseline() -> None:
    """Verifies that direct extraction starts at a higher baseline than inferred extraction."""
    scorer = ConfidenceScorer()
    doc_text = "Project: Oakridge Pavilion"

    direct_cand = RawFieldCandidate(
        value="Oakridge Pavilion",
        evidence="Project: Oakridge Pavilion",
        is_inferred=False,
    )
    inferred_cand = RawFieldCandidate(
        value="Oakridge Pavilion",
        evidence="Project: Oakridge Pavilion",
        is_inferred=True,
    )

    score_direct = scorer.compute_field_confidence(
        field_name="project_name",
        value="Oakridge Pavilion",
        raw_candidate=direct_cand,
        document_text=doc_text,
    )
    score_inferred = scorer.compute_field_confidence(
        field_name="project_name",
        value="Oakridge Pavilion",
        raw_candidate=inferred_cand,
        document_text=doc_text,
    )

    assert score_direct > score_inferred


def test_hallucinated_evidence_receives_penalty() -> None:
    """Verifies that fabricated or unverified evidence results in a penalty."""
    scorer = ConfidenceScorer()
    doc_text = "CHANGE ORDER NO: 004"

    # Evidence matches source text
    verified_cand = RawFieldCandidate(
        value="004",
        evidence="CHANGE ORDER NO: 004",
        is_inferred=False,
    )
    # Evidence NOT in source text (hallucinated quote)
    hallucinated_cand = RawFieldCandidate(
        value="004",
        evidence="As confirmed in formal change document CO-004 on page 7",
        is_inferred=False,
    )

    score_verified = scorer.compute_field_confidence(
        field_name="change_order_number",
        value="004",
        raw_candidate=verified_cand,
        document_text=doc_text,
    )
    score_hallucinated = scorer.compute_field_confidence(
        field_name="change_order_number",
        value="004",
        raw_candidate=hallucinated_cand,
        document_text=doc_text,
    )

    assert score_verified > score_hallucinated
    assert score_hallucinated < 0.70


def test_validation_failure_and_conflict_penalties() -> None:
    """Verifies penalties applied when a field fails validation or has conflicts."""
    scorer = ConfidenceScorer()
    doc_text = "Cost Impact: $15,000.00"
    cand = RawFieldCandidate(
        value="15000",
        evidence="Cost Impact: $15,000.00",
        is_inferred=False,
    )

    score_valid = scorer.compute_field_confidence(
        field_name="cost_impact",
        value="15000",
        raw_candidate=cand,
        document_text=doc_text,
        is_valid=True,
        has_conflict=False,
    )

    score_invalid_and_conflicted = scorer.compute_field_confidence(
        field_name="cost_impact",
        value="15000",
        raw_candidate=cand,
        document_text=doc_text,
        is_valid=False,
        has_conflict=True,
    )

    assert score_valid > score_invalid_and_conflicted


def test_ocr_degradation_penalty() -> None:
    """Verifies that low OCR confidence decreases field confidence score."""
    scorer = ConfidenceScorer()
    doc_text = "Date: 2024-05-15"
    cand = RawFieldCandidate(
        value="2024-05-15",
        evidence="Date: 2024-05-15",
        is_inferred=False,
    )

    score_high_ocr = scorer.compute_field_confidence(
        field_name="issue_date",
        value="2024-05-15",
        raw_candidate=cand,
        document_text=doc_text,
        ocr_confidence=0.95,
    )

    score_low_ocr = scorer.compute_field_confidence(
        field_name="issue_date",
        value="2024-05-15",
        raw_candidate=cand,
        document_text=doc_text,
        ocr_confidence=0.45,
    )

    assert score_high_ocr > score_low_ocr
