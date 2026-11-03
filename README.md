# Construction Change-Order Extraction Pipeline

A clean, reliable Python application for extracting structured information, verifiable evidence, and field-level confidence scores from construction change-order documents in PDF or raw-text format.

Built as an engineering assessment for Sledge.

---

## Features

- **Document Ingestion**: Supports direct text payloads and multi-page PDF document uploads.
- **Hybrid Extraction**: High-speed digital text extraction via PyMuPDF with automatic OCR fallback using Tesseract OCR and Pillow for scanned or image-based documents.
- **Structure-Preserving Normalization**: Standardizes Unicode artifacts and line breaks while preserving tables and page demarcations (`--- [Page X] ---`).
- **Structured LLM Extraction**: OpenAI structured outputs (`client.beta.chat.completions.parse`) with Pydantic v2 schemas. No brittle regex parsing of conversational text.
- **Evidence Citations with Page References**: Every extracted field includes verbatim source text evidence and page number citations.
- **Heuristic Confidence Scoring**: Deterministic multi-factor scoring based on evidence presence, verbatim verification, clarity anchors, validation results, and OCR quality metrics. Missing fields strictly receive `0.0`.
- **Data Validation & Consistency**: Validates dates, numeric values, currencies, positive/negative additions vs. deductions, and checks contract sum arithmetic consistency ($\text{Previous} + \text{Cost} = \text{Revised}$). Flags discrepancies without silently altering source evidence.
- **Production Guardrails**: File size upload limits (10 MB), PDF page limits (50 pages), LLM timeouts, structured logging without sensitive data leakage, and comprehensive error handling.

---

## 15 Extracted Fields

| Field Name | Type | Description |
|---|---|---|
| `project_name` | `string` or `null` | Name of the construction project |
| `project_address` | `string` or `null` | Physical site address |
| `change_order_number` | `string` or `null` | Unique identifier (e.g. `CO-007`) |
| `original_contract_number` | `string` or `null` | Base contract reference (e.g. `CNT-2023-8891`) |
| `title` | `string` or `null` | Short title or subject of the change |
| `description` | `string` or `null` | Detailed narrative of changed scope |
| `contractor_name` | `string` or `null` | Prime contractor entity name |
| `owner_name` | `string` or `null` | Project owner or client entity name |
| `issue_date` | `string` (ISO `YYYY-MM-DD`) or `null` | Date of issuance |
| `requested_by` | `string` or `null` | Entity or person initiating the change |
| `reason_for_change` | `string` or `null` | Justification (e.g., unforeseen conditions, design error) |
| `cost_impact` | `CostImpact` or `null` | Financial change: amount, currency, and type (`addition`, `deduction`, `no_cost`) |
| `schedule_impact_days` | `integer` or `null` | Calendar/working days (+ delay, - acceleration, 0 no impact) |
| `revised_contract_value` | `float` or `null` | New total contract sum after change order |
| `approval_status` | `ApprovalStatus` or `null` | Status (`approved`, `pending`, `rejected`, `draft`, `unknown`) |

Every field is wrapped in:
```json
{
  "value": ...,
  "confidence": 0.95,
  "evidence": "Exact source quote...",
  "page": 1
}
```

---

## Requirements

- Python 3.11+ (tested on Python 3.12)
- (Optional for OCR) Tesseract OCR engine installed on system:
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Install [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

---

## Installation

1. **Clone repository and enter directory:**
   ```bash
   git clone https://github.com/abdulwaheed-aiengineer/sledge-change-order-extractor.git
   cd sledge-change-order-extractor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install application and dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

---

## Environment Setup

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Configure your environment variables in `.env`:
```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=60.0
MAX_UPLOAD_SIZE_BYTES=10485760
MAX_PDF_PAGES=50
OCR_MIN_CHAR_DENSITY=50
LOG_LEVEL=INFO
```

---

## Running the Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation (Swagger UI) is available at:
`http://localhost:8000/docs`

---

## API Usage & Examples

### 1. Health Diagnostic Check
```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "ocr_available": true,
  "llm_configured": true,
  "max_upload_size_bytes": 10485760,
  "max_pdf_pages": 50
}
```

### 2. Extract from Raw Text (`POST /extract/text`)
```bash
curl -X POST http://localhost:8000/extract/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "PROJECT: Oakridge Pavilion\nCHANGE ORDER: CO-007\nDate: 2024-05-15\nCost Impact: The Contract Sum will be increased by $14,500.00\nSchedule: +7 calendar days\nSTATUS: Approved"
  }'
```

### 3. Extract from PDF (`POST /extract/pdf`)
```bash
curl -X POST http://localhost:8000/extract/pdf \
  -F "file=@tests/samples/change_order_1_standard.pdf"
```

**Example JSON Response:**
```json
{
  "document_status": "success",
  "source_type": "digital_pdf",
  "extraction_warnings": [],
  "project_name": {
    "value": "Oakridge Medical Pavilion",
    "confidence": 0.95,
    "evidence": "PROJECT: Oakridge Medical Pavilion",
    "page": 1
  },
  "project_address": {
    "value": "1044 Highland Parkway, Denver, CO 80202",
    "confidence": 0.95,
    "evidence": "1044 Highland Parkway, Denver, CO 80202",
    "page": 1
  },
  "change_order_number": {
    "value": "CO-007",
    "confidence": 1.0,
    "evidence": "CHANGE ORDER NUMBER: CO-007",
    "page": 1
  },
  "original_contract_number": {
    "value": "CNT-2023-8891",
    "confidence": 1.0,
    "evidence": "ORIGINAL CONTRACT NUMBER: CNT-2023-8891",
    "page": 1
  },
  "title": {
    "value": "Electrical conduit rerouting due to unforeseen structural grade beam",
    "confidence": 0.95,
    "evidence": "Title: Electrical conduit rerouting due to unforeseen structural grade beam",
    "page": 1
  },
  "description": {
    "value": "Reroute electrical power feeds and branch circuits around newly exposed subsurface grade beam discovered during basement slab excavation...",
    "confidence": 0.95,
    "evidence": "Description: Reroute electrical power feeds...",
    "page": 1
  },
  "contractor_name": {
    "value": "Apex Commercial Builders LLC",
    "confidence": 0.95,
    "evidence": "CONTRACTOR: Apex Commercial Builders LLC",
    "page": 1
  },
  "owner_name": {
    "value": "Colorado Health Properties Inc.",
    "confidence": 0.95,
    "evidence": "OWNER: Colorado Health Properties Inc.",
    "page": 1
  },
  "issue_date": {
    "value": "2024-05-15",
    "confidence": 1.0,
    "evidence": "DATE OF ISSUANCE: 2024-05-15",
    "page": 1
  },
  "requested_by": {
    "value": "Structural Engineer (Thornton Tomasetti)",
    "confidence": 0.95,
    "evidence": "Requested by: Structural Engineer (Thornton Tomasetti)",
    "page": 1
  },
  "reason_for_change": {
    "value": "Unforeseen existing structural conditions not shown on civil record drawings.",
    "confidence": 0.95,
    "evidence": "Reason for change: Unforeseen existing structural conditions...",
    "page": 1
  },
  "cost_impact": {
    "value": {
      "amount": 14500.0,
      "currency": "USD",
      "impact_type": "addition"
    },
    "confidence": 1.0,
    "evidence": "The Contract Sum will be increased by this Change Order in the amount of: $14,500.00",
    "page": 1
  },
  "schedule_impact_days": {
    "value": 7,
    "confidence": 1.0,
    "evidence": "The Contract Time will be increased by: 7 calendar days",
    "page": 1
  },
  "revised_contract_value": {
    "value": 1309500.0,
    "confidence": 1.0,
    "evidence": "The new Contract Sum including this Change Order will be: $1,309,500.00",
    "page": 1
  },
  "approval_status": {
    "value": "approved",
    "confidence": 1.0,
    "evidence": "STATUS: Fully Executed and Approved",
    "page": 1
  }
}
```

---

## Running Tests & Code Quality

Run the Pytest suite (mocked external calls allow running without an API key):

```bash
pytest -v
```

Run Ruff linter and formatter:

```bash
ruff check .
ruff format --check .
```

---

## Known Limitations

1. **Complex Embedded Multi-Column Tables**: While PyMuPDF layout parsing preserves sequential line blocks, heavily nested multi-column tables with merged cells without borders may require vision-based layout analysis.
2. **Handwritten Field Extraction**: While Tesseract OCR extracts printed text with high fidelity, handwritten cursive remarks in change order margins have variable OCR accuracy; vision LLMs (multimodal OCR) would be required for handwritten markups.
3. **Multi-Currency Conversion**: When a change order lists items in mixed currencies (e.g. CAD and USD on cross-border projects), the pipeline captures the stated currency code but does not perform currency conversion.
4. **Heuristic Confidence vs. Calibrated Probability**: Confidence scores reflect evidence presence, verbatim verification, conflict detection, and validation checks. They are deterministic engineering heuristics, not Bayesian probabilities.
