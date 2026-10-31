"""Validation service for construction change-order data integrity and consistency."""

import logging
import re
from datetime import date, datetime
from typing import NamedTuple

from app.models.change_order import RawChangeOrderExtraction
from app.models.enums import ApprovalStatus, CostImpactType
from app.models.extracted_field import CostImpact

logger = logging.getLogger("sledge.extractor.validator")


class ValidationResult(NamedTuple):
    """Container for validated outputs, warnings, and field-level validation status."""

    normalized_date: str | None
    normalized_cost: CostImpact | None
    normalized_schedule_days: int | None
    normalized_revised_contract: float | None
    normalized_approval_status: ApprovalStatus | None
    field_validation_flags: dict[str, bool]  # field_name -> True if valid, False if failed
    warnings: list[str]


class ChangeOrderValidator:
    """Validates date formats, numeric impacts, contract arithmetic consistency, and conflicts."""

    @staticmethod
    def validate_date(raw_date_str: str | None) -> tuple[str | None, bool, str | None]:
        """Validates and standardizes dates to ISO YYYY-MM-DD format.

        Returns:
            Tuple of (iso_date_string, is_valid, warning_message).
        """
        if not raw_date_str:
            return None, True, None

        cleaned = raw_date_str.strip()

        # Try standard ISO format first
        iso_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", cleaned)
        if iso_match:
            year, month, day = map(int, iso_match.groups())
            try:
                parsed_date = date(year, month, day)
                if not (1970 <= parsed_date.year <= 2050):
                    return (
                        parsed_date.isoformat(),
                        False,
                        f"Date year {parsed_date.year} is outside plausible range (1970-2050).",
                    )
                return parsed_date.isoformat(), True, None
            except ValueError as exc:
                return cleaned, False, f"Invalid calendar date '{cleaned}': {exc}"

        # Common construction date formats: MM/DD/YYYY, MM-DD-YYYY, Month DD, YYYY
        common_formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y",
            "%Y/%m/%d",
        ]
        for fmt in common_formats:
            try:
                dt = datetime.strptime(cleaned, fmt)
                if not (1970 <= dt.year <= 2050):
                    return (
                        dt.strftime("%Y-%m-%d"),
                        False,
                        f"Date year {dt.year} is outside plausible range (1970-2050).",
                    )
                return dt.strftime("%Y-%m-%d"), True, None
            except ValueError:
                continue

        # Date could not be parsed into a calendar date
        return cleaned, False, f"Unrecognized date format for issue_date: '{cleaned}'"

    @staticmethod
    def validate_cost_impact(
        raw_amount: float | None,
        impact_type: CostImpactType,
        currency: str = "USD",
    ) -> tuple[CostImpact | None, bool, str | None]:
        """Validates currency amount and aligns positive/negative signs with impact type.

        Returns:
            Tuple of (CostImpact, is_valid, warning_message).
        """
        if raw_amount is None:
            return None, True, None

        clean_currency = currency.upper().strip() if currency else "USD"

        # Check for No-Cost changes
        if impact_type == CostImpactType.NO_COST:
            if abs(raw_amount) > 0.001:
                return (
                    CostImpact(amount=0.0, currency=clean_currency, impact_type=impact_type),
                    False,
                    f"Cost impact marked as no_cost but non-zero amount ({raw_amount}) was given.",
                )
            return (
                CostImpact(amount=0.0, currency=clean_currency, impact_type=impact_type),
                True,
                None,
            )

        # Check for Deductions (must be negative or zero)
        if impact_type == CostImpactType.DEDUCTION:
            # Normalize to negative number
            normalized_amount = -abs(raw_amount)
            return (
                CostImpact(
                    amount=normalized_amount,
                    currency=clean_currency,
                    impact_type=impact_type,
                ),
                True,
                None,
            )

        # Check for Additions (must be positive)
        if impact_type == CostImpactType.ADDITION:
            if raw_amount < 0:
                return (
                    CostImpact(
                        amount=raw_amount,
                        currency=clean_currency,
                        impact_type=impact_type,
                    ),
                    False,
                    f"Cost impact marked as addition but has negative value ({raw_amount}).",
                )
            return (
                CostImpact(
                    amount=abs(raw_amount),
                    currency=clean_currency,
                    impact_type=impact_type,
                ),
                True,
                None,
            )

        return None, False, f"Unknown cost impact type: {impact_type}"

    @staticmethod
    def validate_schedule_impact(raw_days: int | None) -> tuple[int | None, bool, str | None]:
        """Validates calendar or working days schedule adjustment."""
        if raw_days is None:
            return None, True, None

        if abs(raw_days) > 3650:  # Exceeds 10 years
            return (
                raw_days,
                False,
                f"Schedule impact of {raw_days} days is outside expected range (< 3650 days).",
            )

        return raw_days, True, None

    @staticmethod
    def validate_revised_contract_value(
        raw_val: str | None,
    ) -> tuple[float | None, bool, str | None]:
        """Parses and validates revised contract value."""
        if not raw_val:
            return None, True, None

        cleaned = str(raw_val).replace("$", "").replace(",", "").strip()
        try:
            val = float(cleaned)
            if val < 0:
                return (
                    val,
                    False,
                    f"Revised contract value cannot be negative: {val}",
                )
            return val, True, None
        except ValueError:
            return None, False, f"Could not parse revised contract value '{raw_val}' as a float."

    @staticmethod
    def validate_approval_status(
        raw_status_str: str | None,
    ) -> tuple[ApprovalStatus | None, bool, str | None]:
        """Normalizes workflow approval status string to enum."""
        if not raw_status_str:
            return None, True, None

        cleaned = raw_status_str.strip().lower()
        if "pend" in cleaned or "proposed" in cleaned or "await" in cleaned:
            return ApprovalStatus.PENDING, True, None
        if "draft" in cleaned:
            return ApprovalStatus.DRAFT, True, None
        if "reject" in cleaned or "denied" in cleaned or "disapprov" in cleaned:
            return ApprovalStatus.REJECTED, True, None
        if "approv" in cleaned or "executed" in cleaned or "signed" in cleaned:
            return ApprovalStatus.APPROVED, True, None

        return (
            ApprovalStatus.UNKNOWN,
            True,
            f"Approval status '{raw_status_str}' classified as unknown.",
        )

    def validate_extraction(self, raw: RawChangeOrderExtraction) -> ValidationResult:
        """Runs comprehensive validation and consistency verification on extracted fields."""
        warnings: list[str] = []
        flags: dict[str, bool] = {}

        # 1. Date Validation
        norm_date, date_valid, date_warn = self.validate_date(raw.issue_date.value)
        flags["issue_date"] = date_valid
        if date_warn:
            warnings.append(date_warn)

        # 2. Cost Impact Validation
        norm_cost, cost_valid, cost_warn = self.validate_cost_impact(
            raw_amount=raw.cost_impact.amount,
            impact_type=raw.cost_impact.impact_type,
            currency=raw.cost_impact.currency,
        )
        flags["cost_impact"] = cost_valid
        if cost_warn:
            warnings.append(cost_warn)

        # 3. Schedule Impact Validation
        norm_schedule, sched_valid, sched_warn = self.validate_schedule_impact(
            raw.schedule_impact_days.days
        )
        flags["schedule_impact_days"] = sched_valid
        if sched_warn:
            warnings.append(sched_warn)

        # 4. Revised Contract Value Validation
        norm_revised, revised_valid, revised_warn = self.validate_revised_contract_value(
            raw.revised_contract_value.value
        )
        flags["revised_contract_value"] = revised_valid
        if revised_warn:
            warnings.append(revised_warn)

        # 5. Approval Status Validation
        norm_approval, app_valid, app_warn = self.validate_approval_status(
            raw.approval_status.value
        )
        flags["approval_status"] = app_valid
        if app_warn:
            warnings.append(app_warn)

        # 6. Contract Value Arithmetic Consistency Check
        # Check: previous_contract_sum + cost_impact == revised_contract_value
        if (
            raw.previous_contract_sum is not None
            and norm_cost is not None
            and norm_revised is not None
        ):
            expected_revised = raw.previous_contract_sum + norm_cost.amount
            difference = abs(expected_revised - norm_revised)
            if difference > 1.0:  # Allow $1 tolerance for penny rounding
                arithmetic_warn = (
                    f"Contract arithmetic discrepancy: previous contract sum (${raw.previous_contract_sum:,.2f}) "
                    f"+ cost impact (${norm_cost.amount:,.2f}) = expected ${expected_revised:,.2f}, "
                    f"but revised contract value is stated as ${norm_revised:,.2f} (difference: ${difference:,.2f})."
                )
                warnings.append(arithmetic_warn)
                flags["revised_contract_value"] = False
                flags["cost_impact_consistency"] = False

        # 7. Document Conflicts
        if raw.detected_conflicts:
            for conflict in raw.detected_conflicts:
                conflict_msg = f"Document conflict detected: {conflict}"
                warnings.append(conflict_msg)

        # Default string fields as valid if non-empty or properly missing
        for str_field in [
            "project_name",
            "project_address",
            "change_order_number",
            "original_contract_number",
            "title",
            "description",
            "contractor_name",
            "owner_name",
            "requested_by",
            "reason_for_change",
        ]:
            raw_field = getattr(raw, str_field)
            flags[str_field] = bool(raw_field.value is None or len(str_field) > 0)

        return ValidationResult(
            normalized_date=norm_date,
            normalized_cost=norm_cost,
            normalized_schedule_days=norm_schedule,
            normalized_revised_contract=norm_revised,
            normalized_approval_status=norm_approval,
            field_validation_flags=flags,
            warnings=warnings,
        )
