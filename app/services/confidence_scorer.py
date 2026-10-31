"""Documented heuristic confidence scoring engine.

These confidence scores are deterministic heuristic estimates reflecting evidence
presence, validation results, and document clarity, and are not calibrated statistical
probabilities.
"""

import logging
import re
from typing import Any

from app.models.change_order import (
    RawCostImpactCandidate,
    RawFieldCandidate,
    RawScheduleImpactCandidate,
)

logger = logging.getLogger("sledge.extractor.confidence")


class ConfidenceScorer:
    """Calculates field-level heuristic confidence scores based on empirical document metrics."""

    @staticmethod
    def _verify_evidence_in_text(evidence: str | None, document_text: str) -> tuple[bool, bool]:
        """Checks if quoted evidence is genuinely present in the source text.

        Returns:
            Tuple of (exact_match, partial_match).
        """
        if not evidence or not evidence.strip():
            return False, False

        clean_evidence = re.sub(r"\s+", " ", evidence.strip().lower())
        clean_doc = re.sub(r"\s+", " ", document_text.lower())

        if clean_evidence in clean_doc:
            return True, True

        # Check partial overlap (token-based match for OCR or line-wrapped phrases)
        tokens = [t for t in re.split(r"\W+", clean_evidence) if len(t) > 2]
        if not tokens:
            return False, False

        matched_tokens = sum(1 for t in tokens if t in clean_doc)
        token_ratio = matched_tokens / len(tokens)
        return False, token_ratio >= 0.65

    @staticmethod
    def _check_evidence_clarity(field_name: str, evidence: str | None) -> float:
        """Evaluates whether evidence contains concise, unambiguous field anchors."""
        if not evidence:
            return 0.0

        ev_lower = evidence.lower()
        anchors = {
            "change_order_number": ["change order", "co #", "co-", "c.o.", "order no", "c.o.#"],
            "original_contract_number": [
                "contract #",
                "contract no",
                "agreement #",
                "contract date",
            ],
            "issue_date": ["date", "dated", "/", "-"],
            "cost_impact": [
                "$",
                "cost",
                "sum",
                "increase",
                "decrease",
                "addition",
                "deduct",
                "total",
            ],
            "schedule_impact_days": [
                "day",
                "schedule",
                "calendar days",
                "substantial completion",
                "time",
            ],
            "revised_contract_value": ["new contract", "revised", "contract sum", "new total"],
            "approval_status": [
                "approved",
                "signature",
                "accepted",
                "signed",
                "pending",
                "rejected",
            ],
        }

        field_anchors = anchors.get(field_name, [])
        if any(anchor in ev_lower for anchor in field_anchors):
            return 0.05

        # Penalty if evidence is overly verbose (> 150 words) without clear label
        word_count = len(evidence.split())
        if word_count > 150:
            return -0.05

        return 0.0

    def compute_field_confidence(
        self,
        field_name: str,
        value: Any,
        raw_candidate: RawFieldCandidate | RawCostImpactCandidate | RawScheduleImpactCandidate,
        document_text: str,
        is_valid: bool = True,
        has_conflict: bool = False,
        ocr_confidence: float | None = None,
    ) -> float:
        """Calculates a heuristic confidence score between 0.0 and 1.0.

        Scoring Rules:
        - If value is None: 0.0 (strictly 0.0 for missing values).
        - Direct extraction baseline: 0.85
        - Inferred extraction baseline: 0.65
        - Evidence verified verbatim in document text: +0.05
        - Evidence partially matched in text: 0.0
        - Evidence completely unverified / fabricated: -0.30
        - Value present but evidence omitted: -0.25
        - Field clarity / explicit anchor label: +0.05
        - Validation failed or arithmetic discrepancy: -0.25
        - Field conflicts with another document statement: -0.20
        - OCR degradation penalty: -0.05 to -0.25 if OCR confidence is low
        """
        # Rule: Missing fields must not receive artificially high confidence
        if value is None:
            return 0.0

        is_inferred = getattr(raw_candidate, "is_inferred", False)
        base = 0.65 if is_inferred else 0.85

        score = base

        # 1. Evidence availability & verification
        evidence = getattr(raw_candidate, "evidence", None)
        if evidence and evidence.strip():
            exact, partial = self._verify_evidence_in_text(evidence, document_text)
            if exact:
                score += 0.05
            elif partial:
                score += 0.00
            else:
                # Evidence could not be located in source document
                score -= 0.30
        else:
            # Value provided without supporting evidence
            score -= 0.25

        # 2. Evidence clarity & label presence
        score += self._check_evidence_clarity(field_name, evidence)

        # 3. Validation results
        if not is_valid:
            score -= 0.25
        else:
            score += 0.05

        # 4. Conflicting values penalty
        if has_conflict:
            score -= 0.20

        # 5. OCR Quality penalty where measurable
        if ocr_confidence is not None:
            if ocr_confidence >= 0.90:
                pass  # High OCR quality
            elif ocr_confidence >= 0.70:
                score -= 0.05
            elif ocr_confidence >= 0.50:
                score -= 0.15
            else:
                score -= 0.25

        # Clamp between 0.05 and 1.0 for non-null values
        return max(0.05, min(1.0, round(score, 2)))
