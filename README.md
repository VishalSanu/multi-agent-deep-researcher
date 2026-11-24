# Multi-Agent Deep Researcher 🔬

A high-rigor, asynchronous multi-agent research engine built with Python, Pydantic, and the Google GenAI SDK (`google-genai`).

The system deploys a coordinated triad of specialized agents that investigate complex topics, extract verifiable claims from primary web sources, detect and arbitrate conflicting empirical assertions, and compile publication-grade cited briefs.

---

## 🏛️ Agent Architecture & Workflow

```
[User Topic / Query]
         │
         ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. SEARCHER AGENT (gemini-2.5-flash)                   │
 │    • Deconstructs topic into 3-5 high-precision queries │
 │    • Searches via Tavily (or DuckDuckGo fallback)       │
 │    • Scrapes primary sources (.gov, .edu, arXiv, docs)  │
 │      via Trafilatura & httpx                            │
 └─────────────────────────┬───────────────────────────────┘
                           │ Scraped Documents
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 2. FACT-CHECKER AGENT (gemini-2.5-flash + gemini-3.1-pro-preview)   │
 │    • Extracts atomic claims with exact verbatim quotes  │
 │    • Builds cross-document claim & credibility matrix   │
 │    • Detects empirical contradictions & nuances         │
 └─────────────────────────┬───────────────────────────────┘
                           │
             Contradictions requiring proof?
                ├── Yes (<= max loops) ──> [Targeted Re-Search Loop]
                └── No / Resolved
                           │ Verified Claims & Conflicts
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 3. WRITER AGENT (gemini-3.1-pro-preview)                │
 │    • Maps sources to strict numbered citations [1]      │
 │    • Drafts Executive Summary & Thematic Syntheses      │
 │    • Constructs "Conflicting Claims & Nuances" Table    │
 │    • Generates Annotated Bibliography with archival     │
 │      quotes and domain authority ratings                │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
              [Final Cited Brief (.md / .json)]
```

---

## 🚀 Quickstart

### 1. Requirements & Installation

Python 3.10+ is supported. Install dependencies:

```bash
pip install -r requirements.txt
# Or using uv for instant installation:
uv pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Add your Gemini API key:

```env
GEMINI_API_KEY="your-gemini-api-key"

# Optional: Tavily API key for enhanced search.
# If omitted, DuckDuckGo search is used automatically.
TAVILY_API_KEY="your-tavily-api-key"
```

### 3. Launching the Interactive Web UI 🎨

Launch the Streamlit research studio:

```bash
.\.venv\Scripts\python -m streamlit run app.py
```
*(Or if running in an activated venv: `streamlit run app.py`)*

The UI opens at `http://localhost:8501`, offering:
- **API Key Management**: Enter and validate your Gemini and Tavily keys, with an optional one-click save to `.env`.
- **Live Multi-Agent Stepper**: Watch the Searcher, Fact-Checker, and Writer agents execute and resolve conflicts in real-time.
- **Tabbed Results Dashboard**:
  - 📄 **Executive Brief**: Formatted synthesis with numbered citations `[1]`.
  - ⚖️ **Conflicting Claims & Nuances**: Interactive dispute comparison table.
  - 📚 **Source Bibliography**: Domain authority ratings & archival excerpts.
  - 🔍 **Claim Audit Matrix**: Granular atomic claims data table.
- **Instant Export**: Download `.md` narrative brief or structured `.json` claim matrix.
- **History Explorer**: Revisit and reload previous research sessions stored locally in `history/`.

### 4. Running Research via CLI

Run a deep research investigation:

```bash
python -m deep_researcher.main "Impact of Solid-State Batteries on Grid-Scale Storage vs Lithium-Iron-Phosphate"
```

With custom flags:

```bash
python -m deep_researcher.main \
  --topic "mRNA Cancer Vaccine Clinical Trial Outcomes: Efficacy vs Long-term Immune Tolerance" \
  --max-sources 12 \
  --max-search-loops 2 \
  --output "mrna_cancer_vaccine_brief.md" \
  --save-json
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `topic` / `--topic` | _Required_ | The complex topic, hypothesis, or policy question |
| `-s`, `--max-sources` | `10` | Maximum candidate primary sources to scrape |
| `-l`, `--max-search-loops` | `2` | Max targeted search iterations to resolve contested claims |
| `-o`, `--output` | `deep_research_brief_<slug>.md` | Destination file for the markdown brief |
| `--model-fast` | `gemini-2.5-flash` | Model for query planning & claim extraction |
| `--model-pro` | `gemini-3.1-pro-preview` | Model for conflict arbitration & final synthesis |
| `--save-json` | `False` | Also outputs full structured claim matrix & bibliography as JSON |

---

## 📦 Project Structure

```
deep_researcher/
├── __init__.py
├── main.py              # Rich CLI runner & terminal reporting
├── orchestrator.py      # Async state machine and multi-agent coordination loop
├── models/
│   ├── __init__.py
│   └── schemas.py       # Pydantic schemas (Claims, Conflicts, Sources, Brief)
├── agents/
│   ├── __init__.py
│   ├── base.py          # Async Google GenAI SDK wrapper
│   ├── searcher.py      # Query planning, search execution, and scraper
│   ├── fact_checker.py  # Atomic claim extraction & conflict detection
│   └── writer.py        # Publication-grade synthesis & citation mapping
└── tools/
    ├── __init__.py
    ├── search.py        # Tavily & DuckDuckGo hybrid search
    └── scraper.py       # Trafilatura + httpx content extraction & domain evaluator
```
