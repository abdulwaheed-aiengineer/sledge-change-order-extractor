"""Tests for data validation and consistency verification."""

from app.models.change_order import (
    RawChangeOrderExtraction,
    RawCostImpactCandidate,
    RawFieldCandidate,
)
from app.models.enums import ApprovalStatus, CostImpactType
from app.services.validator import ChangeOrderValidator


def test_validate_date_formats() -> None:
    """Verifies standardizing ISO and common non-ISO construction date formats."""
    validator = ChangeOrderValidator()

    # Standard ISO
    iso_val, is_valid, warn = validator.validate_date("2024-05-15")
    assert is_valid
    assert iso_val == "2024-05-15"
    assert warn is None

    # MM/DD/YYYY
    us_val, is_valid, warn = validator.validate_date("05/15/2024")
    assert is_valid
    assert us_val == "2024-05-15"

    # Text format
    txt_val, is_valid, warn = validator.validate_date("October 12, 2024")
    assert is_valid
    assert txt_val == "2024-10-12"


def test_validate_invalid_date() -> None:
    """Verifies that impossible calendar dates (like Feb 30) produce warnings."""
    validator = ChangeOrderValidator()
    val, is_valid, warn = validator.validate_date("2024-02-30")

    assert not is_valid
    assert warn is not None
    assert "Invalid calendar date" in warn or "Unrecognized date format" in warn


def test_validate_cost_impact_addition() -> None:
    """Verifies addition cost impact validation."""
    validator = ChangeOrderValidator()
    cost, is_valid, warn = validator.validate_cost_impact(
        raw_amount=14500.0,
        impact_type=CostImpactType.ADDITION,
        currency="USD",
    )

    assert is_valid
    assert cost is not None
    assert cost.amount == 14500.0
    assert cost.impact_type == CostImpactType.ADDITION
    assert cost.currency == "USD"
    assert warn is None


def test_validate_cost_impact_deduction() -> None:
    """Verifies deduction cost impact normalizes to a negative amount."""
    validator = ChangeOrderValidator()
    cost, is_valid, warn = validator.validate_cost_impact(
        raw_amount=32500.0,  # Presented as positive in source text
        impact_type=CostImpactType.DEDUCTION,
        currency="USD",
    )

    assert is_valid
    assert cost is not None
    assert cost.amount == -32500.0  # Normalized to negative
    assert cost.impact_type == CostImpactType.DEDUCTION


def test_validate_cost_impact_no_cost() -> None:
    """Verifies no-cost change impact."""
    validator = ChangeOrderValidator()
    cost, is_valid, warn = validator.validate_cost_impact(
        raw_amount=0.0,
        impact_type=CostImpactType.NO_COST,
        currency="USD",
    )

    assert is_valid
    assert cost is not None
    assert cost.amount == 0.0
    assert cost.impact_type == CostImpactType.NO_COST


def test_contract_value_arithmetic_consistency_check() -> None:
    """Verifies that arithmetic consistency between previous contract, cost, and revised value is validated."""
    validator = ChangeOrderValidator()

    # Case 1: Consistent arithmetic ($1,000,000 + $50,000 = $1,050,000)
    raw_consistent = RawChangeOrderExtraction(
        cost_impact=RawCostImpactCandidate(
            amount=50000.0,
            currency="USD",
            impact_type=CostImpactType.ADDITION,
        ),
        revised_contract_value=RawFieldCandidate(value="1050000.00"),
        previous_contract_sum=1000000.0,
    )
    result = validator.validate_extraction(raw_consistent)
    assert result.field_validation_flags.get("revised_contract_value") is True
    assert not any("discrepancy" in w.lower() for w in result.warnings)

    # Case 2: Arithmetic discrepancy ($1,000,000 + $50,000 != $1,200,000)
    raw_inconsistent = RawChangeOrderExtraction(
        cost_impact=RawCostImpactCandidate(
            amount=50000.0,
            currency="USD",
            impact_type=CostImpactType.ADDITION,
        ),
        revised_contract_value=RawFieldCandidate(value="1200000.00"),
        previous_contract_sum=1000000.0,
    )
    res_inconsistent = validator.validate_extraction(raw_inconsistent)
    assert res_inconsistent.field_validation_flags.get("revised_contract_value") is False
    assert any("Contract arithmetic discrepancy" in w for w in res_inconsistent.warnings)


def test_validate_approval_status_mapping() -> None:
    """Verifies mapping natural language approval statements to standard status enums."""
    validator = ChangeOrderValidator()

    status, valid, _ = validator.validate_approval_status("Executed and Signed")
    assert status == ApprovalStatus.APPROVED
    assert valid

    status, valid, _ = validator.validate_approval_status("Awaiting Owner Approval")
    assert status == ApprovalStatus.PENDING
    assert valid

    status, valid, _ = validator.validate_approval_status("Rejected per Architect review")
    assert status == ApprovalStatus.REJECTED
    assert valid
