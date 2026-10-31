# Engineering Approach: Construction Change-Order Extraction Pipeline

## 1. Problem Understanding

Construction change orders (such as AIA G701, AGC, EJCDC, or custom company forms) represent legally binding modifications to the contract scope, cost, and schedule. Extracting information from these documents presents unique domain challenges:

1. **Document Heterogeneity**: Change orders range from cleanly typed digital vector PDFs to scanned, rotated, multi-generation faxes with degraded text and handwritten signatures.
2. **Financial Precision**: Financial extraction must unambiguously distinguish additions, deductions, and no-cost changes. Deductions may be indicated by minus signs, parentheses `($4,500.00)`, words ("deduct", "decrease", "credit"), or separate credit columns.
3. **Contract Math Consistency**: Change order forms frequently summarize:
   $$\text{Previous Contract Sum} + \text{Cost Impact} = \text{New Contract Sum}$$
   Discrepancies in this relationship can indicate typographic errors in the source document, missing items, or extraction errors.
4. **Fidelity over Hallucination**: Downstream ERP and accounting systems cannot tolerate invented contract numbers or hallucinated amounts. Missing fields must remain `null` rather than estimated.
5. **Calibrated Confidence**: LLMs consistently express unwarranted confidence in their own extractions. A transparent, deterministic heuristic is required to score field extraction reliability based on verifiable evidence.

---

## 2. Technical Architecture

The application adopts a modular, pipeline-based architecture with clear separation of concerns and dependency injection:

```
[ PDF Upload / Raw Text ]
           │
           ▼
┌──────────────────────────────────────┐
│  Stage 1: Ingestion & Document Parse │  (PyMuPDF digital text extraction)
└──────────────────┬───────────────────┘
                   │
     [ Low Text Density / Scanned? ]
           ├───► YES ──► [ Stage 2: OCR Fallback Service ] (Tesseract + Pillow)
           └───► NO  ──► [ Stage 3: Text Normalizer ]
                               │
                               ▼
┌──────────────────────────────────────┐
│  Stage 4: LLM Structured Extraction  │  (OpenAI Pydantic structured output)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Stage 5: Business Logic Validation  │  (Dates, signs, arithmetic cross-check)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Stage 6: Heuristic Confidence Score │  (Deterministic multi-factor engine)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Stage 7: Response Serialization     │  (ChangeOrderExtractionResponse JSON)
└──────────────────────────────────────┘
```

### Components

- **`PDFExtractor` (`app/services/pdf_extractor.py`)**: Parses PDF byte streams using PyMuPDF (`pymupdf`). Measures per-page text density and character counts to determine whether OCR is required. Enforces maximum page constraints (default 50 pages) and rejects encrypted documents safely.
- **`OCRService` (`app/services/ocr_service.py`)**: Renders PDF pages to 300 DPI images via Pillow and executes Tesseract OCR. Computes word-level OCR confidence metrics using `image_to_data` and degrades gracefully if Tesseract is not installed.
- **`TextNormalizer` (`app/services/text_normalizer.py`)**: Cleans Unicode artifacts (ligatures, smart quotes, dashes, non-breaking spaces) while preserving layout structure, tables, and page demarcations (`--- [Page X] ---`).
- **`LLMExtractor` (`app/services/llm_extractor.py`)**: Drives OpenAI structured output parsing (`client.beta.chat.completions.parse`) using a separately maintained prompt in `app/prompts/extraction_prompt.py`. Guarantees schema adherence without fragile regex parsing.
- **`ChangeOrderValidator` (`app/services/validator.py`)**: Enforces date validity (calendar accuracy, YYYY-MM-DD), ensures financial signs align with `CostImpactType` (addition, deduction, no_cost), and verifies arithmetic consistency between previous contract values and revised sums.
- **`ConfidenceScorer` (`app/services/confidence_scorer.py`)**: Calculates deterministic field-level confidence scores based on empirical document evidence, validation status, conflict presence, and OCR metrics.
- **`ExtractionPipeline` (`app/services/pipeline.py`)**: Coordinates the sequential workflow and synthesizes the final response.

---

## 3. Extraction Methodology

### Explicit Schema Enforcement
Rather than prompting an LLM for conversational text or using regex on loose Markdown, extraction uses OpenAI's structured output parser tied to a Pydantic model (`RawChangeOrderExtraction`). This ensures:
- Strict JSON conformism matching the schema.
- Native representation of missing fields as `None` (`null`).
- Dedicated fields for source evidence strings and page numbers.

### Distinguishing Financial Impacts
Change order financials are categorized into an explicit enum:
```python
class CostImpactType(StrEnum):
    ADDITION = "addition"
    DEDUCTION = "deduction"
    NO_COST = "no_cost"
```
The validator guarantees sign consistency:
- **Addition**: Positive float (e.g. `14500.00`).
- **Deduction**: Normalized to a negative float (e.g. `-32500.00`).
- **No-Cost**: Normalized to `0.00`.

---

## 4. Validation Strategy

Validation follows a **non-destructive flagging principle**:
> *Flag conflicting or inconsistent values rather than silently changing them. Preserve the source evidence.*

1. **Date Validation**:
   - Dates are parsed against ISO `YYYY-MM-DD` and common formats (`MM/DD/YYYY`, `Month DD, YYYY`).
   - Dates with impossible calendar days (e.g., February 30) or years outside `1970–2050` are flagged with explicit warnings in `extraction_warnings` rather than silently fabricated.
2. **Financial Consistency & Arithmetic Cross-Checking**:
   - When the document explicitly includes both the prior contract sum and the revised contract sum, the pipeline verifies:
     $$|\text{Previous Contract Sum} + \text{Cost Impact} - \text{Revised Contract Value}| \le 1.00$$
   - If this equality fails, an explicit warning is generated:
     ```
     Contract arithmetic discrepancy: previous contract sum ($500,000.00) + cost impact ($25,000.00) = expected $525,000.00, but revised contract value is stated as $540,000.00 (difference: $15,000.00).
     ```
   - The value is retained as reported by the source document, but its validation status is marked failed and its confidence score is penalized.
3. **Approval Status Mapping**:
   - Maps language to `approved`, `pending`, `rejected`, `draft`, or `unknown`.
   - Ensures negative qualifiers (e.g., "awaiting approval", "pending signature") are prioritized over positive keyword substrings ("approv").

---

## 5. Confidence Scoring Methodology

### Heuristic Philosophy
LLMs frequently report high confidence even when hallucinating. This system implements a documented, deterministic heuristic calculation.

> **Important**: These scores are heuristic estimates designed for ranking and threshold filtering; they are not calibrated Bayesian probabilities.

### Heuristic Scoring Rules

$$\text{Confidence} = \text{Base} + \Delta_{\text{evidence}} + \Delta_{\text{clarity}} + \Delta_{\text{validation}} - \Delta_{\text{conflict}} - \Delta_{\text{ocr}}$$

1. **Missing Values**:
   - If `value is None`: Confidence is strictly **`0.0`**. (Missing fields never receive artificially high confidence).
2. **Extraction Baseline**:
   - Direct extraction (explicitly stated in text): `base = 0.85`
   - Inferred extraction (deduced from context or calculations): `base = 0.65`
3. **Evidence Availability & Verbatim Verification**:
   - Evidence provided AND verified verbatim in source document: `+0.05`
   - Evidence partially matched (OCR token overlap): `+0.00`
   - Evidence hallucinated (cannot be found in source text): **`-0.30`**
   - Value present but evidence omitted: **`-0.25`**
4. **Evidence Clarity & Anchor Labels**:
   - Contains explicit field label (e.g. "CHANGE ORDER NO:", "Contract Sum:", "Date:"): `+0.05`
   - Overly diffuse/vague quote (> 150 words without anchor): `-0.05`
5. **Validation Integrity**:
   - Passed all format and calendar checks: `+0.05`
   - Failed validation or contract arithmetic discrepancy: **`-0.25`**
6. **Conflicting Values**:
   - Document contains contradictory numbers or statements: **`-0.20`**
7. **OCR Quality Metric (where measurable)**:
   - When OCR is used, word-level confidence is aggregated:
     - Mean OCR confidence $\ge 0.90$: No penalty (`0.00`)
     - Mean OCR confidence $0.70 \le c < 0.90$: `-0.05`
     - Mean OCR confidence $0.50 \le c < 0.70$: `-0.15`
     - Mean OCR confidence $< 0.50$: `-0.25`
8. **Final Clamping**:
   - Clamped to $[0.05, 1.00]$ for non-null values (and $0.0$ for nulls).

---

## 6. Failure Modes & Mitigations

| Failure Mode | Root Cause | System Mitigation |
|---|---|---|
| **Scanned / Image PDF** | PDF has no selectable text layer | Detects low character density (< 50 chars/page); automatically triggers 300 DPI image rendering and Tesseract OCR. |
| **Missing Tesseract Binary** | Host environment lacks OCR dependencies | OCRService catches missing executable, adds descriptive warning to `extraction_warnings`, and continues with partial digital text without crashing. |
| **Arithmetic Typo in Document** | Signer entered wrong revised contract sum | Does not alter extracted values; flags discrepancy in `extraction_warnings` and reduces field confidence score. |
| **Prompt Injection / Malicious Document** | Adversarial text inside change order description | Strict Pydantic schema validation isolates LLM output to data fields; inputs are treated as data, not system instructions. |
| **Denial of Service (Oversized PDF)** | Multi-gigabyte file or 1,000-page scan | Request size limit (10 MB) triggers HTTP 413; page count limit (50 pages) triggers HTTP 400. |
| **LLM Provider Outage / Timeout** | OpenAI API connectivity issues | Configurable timeout (default 60s); returns HTTP 504 (timeout) or HTTP 502 (bad gateway) with clean JSON error response. |

---

## 7. Future Improvements

1. **Table Structure Extraction with Spatial Bounding Boxes**: Integrate PyMuPDF's `get_text("words")` with visual bounding boxes to preserve table column alignments before feeding text to the LLM.
2. **Multi-Document Package Processing**: Support bundling change orders with attachments (RFIs, subcontractor quotes, schedule Gantt charts) using multi-stage extraction.
3. **Calibrated Confidence via Fine-Tuning**: Train a regression model on human-verified ground truth extraction datasets to produce statistically calibrated probabilities.
4. **Asynchronous Task Queue (Celery/Temporal)**: For large multi-page scans (100+ pages), process extractions asynchronously with webhook callbacks.
