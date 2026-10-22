# Autonomous Document Agent

A multi-agent, multi-model system that turns one sentence into a finished, professionally
formatted Microsoft Word document. It decides **what** document to write, **plans its own
task list**, writes the sections in parallel, **reviews its own work and fixes it**, and
renders a `.docx`.

```
POST /agent  {"request": "Write a project plan to migrate our CRM to Salesforce in 4 months"}
   -> JSON report (plan, live TODO list, self-review scores, model usage) + downloadable .docx
```

Runs entirely on **free** LLM tiers (Groq + Gemini), spreading work across five models so
no single free-tier limit becomes a bottleneck, with an offline fallback so it never
hard-fails in a demo. Typical run: **13-25 seconds, about 15-25K tokens**.

---

## How it works

```
                 ┌──────────────┐
  request ──────►│  Guardrails  │  length, gibberish, prompt-injection, harmful content
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐   doc type, audience, ASSUMPTIONS for missing info,
                 │   Planner    │   CONFLICTS + how they're resolved, KEY FACTS sheet,
                 └──────┬───────┘   4-8 section outline          [gpt-oss-120b]
                        ▼  one TODO per section (created from the plan)
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐   parallel, round-robin across models
   │ Writer  │     │ Writer  │ ... │ Writer  │   [gemini-3.5-flash-lite | gpt-oss-20b]
   └────┬────┘     └────┬────┘     └────┬────┘   each returns typed blocks:
        └───────────────┼───────────────┘        paragraph | bullets | table
                        ▼
                 ┌──────────────┐   score 1-10 vs. the ORIGINAL request; issues with
                 │   Reviewer   │──┐ severity + every section they affect  [gpt-oss-120b]
                 └──────┬───────┘  │
             score < 8? │          │ yes: agent ADDS "Revise section N" tasks to its own
                        │◄─────────┘ TODO list, fixes only those sections, then VERIFIES
                        ▼            the fixes; a revision that scores lower is discarded
                 ┌──────────────┐
                 │ DOCX renderer│   deterministic: cover, contents, key facts, assumptions,
                 └──────────────┘   styled tables, page numbers, execution-log appendix
```

**Key design idea: the LLM decides *content*, code decides *structure*.**
Every LLM answer must match a Pydantic schema (`Plan`, `SectionContent`, `Review`). The
model never writes formatting, so every document is consistently styled and a bad answer
is caught at the step that produced it, not three steps later.

### The agents

| Agent | Model (default) | Output | Why separate |
|---|---|---|---|
| **Planner** | `gpt-oss-120b` (Groq) | `Plan`: type, audience, assumptions, conflicts, key facts, outline | The key decisions get the strongest model |
| **Writers** | `gemini-3.5-flash-lite` + `gpt-oss-20b`, round-robin | `SectionContent` (typed blocks) | Most of the calls, so they're spread across two companies |
| **Reviewer** | `gpt-oss-120b` (Groq) | `Review`: score, issues with severity | A *different* model from the writers is a more independent critic |

The **orchestrator** (`app/orchestrator.py`) coordinates them and owns the live TODO list.
The list is *not* hard-coded: the planner's outline creates the writing tasks, and the
reviewer's verdict decides whether revision and verification tasks are added at run time.

---

## Mandatory engineering improvement: Reflection / self-check

**What:** after the draft is written, the Reviewer agent scores it against the original
request (coverage, consistency, specificity, professionalism) and returns issues, each with
a severity (critical/major/minor) and **every section it affects**. If the score is below
the threshold (default 8), the agent:
1. adds "Revise section N" tasks to its own TODO list for the critical/major issues,
2. rewrites only those sections, showing each reviser the other sections so cross-section
   contradictions can actually be fixed,
3. runs a **verification review** that checks whether each earlier issue is fixed (instead
   of hunting for fresh nitpicks),
4. keeps the revision only if the score didn't drop. At most 2 rounds.

**Why:** the most common failure of LLM-written documents isn't bad grammar. It's silently
skipping part of the request or contradicting itself ("budget $20k" in one section, "$21k"
in another). A single-pass generator can't see this; a separate reviewer with the original
request in hand can.

**Measured effect (real runs):** 7/10 → 9/10 on the standard request (names in the
timeline didn't match the RACI; 3 related sections fixed together), 5/10 → 9/10 on the
complex request.

### Supporting reliability features
- **Multi-model routing and load balancing:** six free models across two providers; each agent has its own model pool; writers
  rotate round-robin. Free tiers limit each *model* separately (Groq: 8K tokens/minute per
  model), so spreading the work multiplies the usable capacity. Every response reports
  `calls_by_agent` and `tokens_by_model`.
- **Rate-limit cooldowns:** a model that returns 429 is put in cooldown (using the
  provider's `Retry-After` or Gemini's `retryDelay`) and the call moves *immediately* to a
  model that's ready, instead of sleeping. Daily-quota errors rest a model for an hour.
- **Shared key-facts sheet:** the planner fixes budget figures, dates, owners and choices
  once; all parallel writers must use them, which removes most cross-section contradictions.
- **Deterministic fact checks:** code verifies what LLMs get wrong (impossible dates such as
  2027-02-29, table Total rows that do not equal the sum of their items) and sends the exact
  error back for correction before the reviewer ever sees the draft. The same applies to
  plan budget parts that exceed the total, cost tables over budget, and placeholders such as
  `[Your Name]`. Every number in the request is also spelled out by code (10000000 =
  1,00,00,000 = 1 crore = 10 million) so the model never has to count zeros.
- **Locale awareness:** the web page sends the browser language and time zone, so an
  amount with no currency is read in the requester's currency (e.g. rupees for en-IN).
- **JSON self-repair:** if output fails schema validation, the exact validation error is
  sent back to the model, which corrects it.
- **Graceful degradation:** if a section can't be generated, a marked placeholder is
  inserted and the run returns `completed_with_warnings` instead of crashing. The reviewer
  never uses the offline model, because a fake quality score is worse than none.
- **Guardrails:** cheap deterministic checks run *before* any tokens are spent; user text is
  always passed to the LLM as data inside a `<context>` block, never as instructions.

---

## Setup (Windows, macOS, Linux)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Put your free keys in `.env`:

```
GROQ_API_KEY=gsk_...      # https://console.groq.com/keys
GEMINI_API_KEY=...        # https://aistudio.google.com/apikey  (optional but recommended)
```

> No key? It still runs, using the offline mock LLM. Every response says which models were used.

Which model serves which agent is configurable in `.env` (`PLANNER_MODELS`,
`WRITER_MODELS`, `REVIEWER_MODELS`) without touching code.

## Run

```bash
uvicorn app.main:app --reload
```

Then use whichever is easiest:

| Open | What you get |
|---|---|
| **http://localhost:8000** | A guided page: type a request (or click an example) → **Generate** → watch the agent's task list update live → **Download Word file**. No file names to copy. |
| **http://localhost:8000/docs** | Interactive API docs with a how-to at the top. `POST /agent` → *Try it out* → *Execute*, then open `download_url`. |
| `python scripts/demo.py` | Runs both assignment test inputs from the terminal (no server needed). |

### API

| Method | Path | Description |
|---|---|---|
| `POST` | `/agent` | **The assignment endpoint.** Body `{"request": "..."}`. Waits for the run and returns the JSON report. Add `?return_file=true` to get the `.docx` directly. |
| `POST` | `/jobs` | Same pipeline in the background: returns a `job_id` immediately (202). |
| `GET` | `/jobs/{job_id}` | The agent's live task list, then the same result as `/agent` when finished. |
| `GET` | `/documents` | Every generated document, newest first, with download links. |
| `GET` | `/documents/{filename}` | Download a generated document. |
| `GET` | `/health` | Status, the LLM fallback chain, and which models each agent uses. |

```bash
curl -X POST http://localhost:8000/agent -H "Content-Type: application/json" -d @examples/standard_request.json
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/agent -ContentType 'application/json' -InFile examples/standard_request.json
```

Response (abridged; per-model numbers are illustrative):

```json
{
  "status": "completed",
  "summary": "Planned and wrote a 7-section Project Plan ... Self-review scored the draft 7/10; after revising 3 section(s) it scored 9/10.",
  "document": {"filename": "crm_migration_to_salesforce_project_plan_5b5fa6c3.docx", "download_url": "http://localhost:8000/documents/...", "word_count": 1258},
  "plan": {
    "assumptions": ["Project start date is 2026-11-01 and end date is 2027-02-28 ..."],
    "conflicts_resolved": ["..."],
    "key_facts": ["Total budget: $120,000 (licensing $30,000, consulting $40,000, ...)", "Project lead: Alex Morgan, Project Manager"],
    "outline": ["1. Executive Summary", "..."]
  },
  "todo": [{"id": 1, "title": "Validate request and apply guardrails", "agent": "guardrails", "status": "done"}, "..."],
  "review": {"initial_score": 7, "final_score": 9, "revised_sections": ["Risk Management", "Resource Allocation & RACI", "Timeline & Milestones"]},
  "llm": {
    "calls": 14, "retries": 1, "total_tokens": 28827,
    "calls_by_agent": {"planner": {"groq/openai/gpt-oss-120b": 1},
                       "writer": {"groq/openai/gpt-oss-20b": 4, "gemini/gemini-3.5-flash-lite": 6},
                       "reviewer": {"groq/openai/gpt-oss-120b": 2}},
    "tokens_by_model": {"groq/openai/gpt-oss-120b": 11200, "groq/openai/gpt-oss-20b": 7400, "gemini/gemini-3.5-flash-lite": 10227}
  }
}
```

## Test inputs

The two assignment test inputs:

| | Request | What it demonstrates |
|---|---|---|
| **Standard** | `examples/standard_request.json`: CRM to Salesforce project plan, 4 months, 5 engineers, $120k | Clean planning, consistent facts, a self-correction loop |
| **Complex** | `examples/complex_request.json`: "short but cover everything", budget under $20k **and** 24/7 in 12 languages from day one, unsure whether proposal or report | Detects each conflict, proposes a feasible phased plan, chooses the document type itself, states assumptions |

More ready-made examples (also one click away on the web page):

| Example | Shows |
|---|---|
| `meeting_minutes_request.json` | Decisions and an action-item table with owners and deadlines |
| `family_trip_request.json` | A personal plan in rupees (with `locale: en-IN`) whose budget table adds up exactly |
| `onboarding_sop_request.json` | A day-by-day procedure with owners and verification steps |

```bash
python scripts/demo.py                 # the two assignment inputs
python scripts/demo.py all             # every example
python scripts/demo.py family_trip     # just one
```

## Tests

```bash
pytest -q
```

66 tests, fully offline (no API calls): end-to-end API runs, background jobs, the web page, the reflection loop, cross-section
fixes, discarding a revision that lowers the score, placeholder recovery, retry / cooldown /
fallback / JSON self-repair, per-agent routing and load balancing, token accounting, the
HTTP provider against a simulated server (429 + Retry-After, daily quota,
`json_validate_failed`, 401/404), guardrails and path traversal.

## Project layout

```
app/
  main.py                 FastAPI app, routes, error handlers
  jobs.py                 Background jobs with a live task list
  static/index.html       Guided web page (no build step, no external assets)
  orchestrator.py         Planner -> Writers -> Reviewer -> Renderer, live TODO list
  agents/                 planner.py, writer.py, reviewer.py, prompts.py
  llm/                    client.py (routing, cooldowns, retry, repair, fallback),
                          providers.py, factory.py (model pools per agent), mock.py
  rendering/              docx_renderer.py (python-docx + raw OOXML helpers)
  schemas.py              Pydantic contracts for every LLM output and the API
  guardrails.py           Input validation
  trace.py                Per-run TODO list, warnings, calls and tokens per model
tests/                    pytest suite
scripts/demo.py           Runs both test inputs and prints the agent's work
examples/                 The two assignment requests
samples/                  Word documents produced by real runs of both requests
```

## Engineering trade-offs

- **Autonomous planning vs. deterministic workflow.** The LLM chooses the document type,
  sections, facts and whether to revise; the pipeline *shape* is fixed and every output is
  schema-validated, so behaviour stays predictable and testable.
- **Multi-agent, multi-model vs. one big prompt.** More calls than a single prompt, but
  each prompt is small and validated, sections run in parallel, a failure is isolated to
  one section, and load is spread across free-tier limits.
- **Parallel writers vs. coherence.** Parallel is fast but writers can't see each other.
  Mitigated by the shared key-facts sheet, by revisers seeing the other sections, and by
  the reviewer checking cross-section consistency.

## Known limitations

- Review quality depends on the reviewer model; if the strong models are rate-limited the
  fallback reviewer is more lenient.
- Free tiers have daily caps (e.g. `gemini-2.5-flash` allows only 20 requests/day, which is
  why it isn't used). If all hosted models are exhausted, writers fall back to the offline
  model and the response is marked `completed_with_warnings`.

## Scaling further

- Move `/agent` to a job queue (Celery/RQ/Arq + Redis): `POST` returns a job id, clients
  poll or receive a webhook. The orchestrator is already async and stateless per run.
- Share model cooldown state across workers (Redis) so a fleet respects one rate limit.
- Store documents in object storage (S3/GCS) instead of local disk.
- Persist run traces for observability and an evaluation set of review scores.
