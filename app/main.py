"""FastAPI entry point.

Run:   uvicorn app.main:app --reload
Open:  http://localhost:8000        (guided web page)
       http://localhost:8000/docs   (interactive API docs)
"""

from __future__ import annotations

import logging
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.guardrails import GuardrailViolation, validate_request
from app.jobs import JobStore
from app.llm import ChatProvider, LLMClient, LLMUnavailableError
from app.llm.factory import build_providers, build_routes
from app.orchestrator import DocumentAgent
from app.schemas import (
    AgentRequest,
    AgentResponse,
    DocumentListItem,
    ErrorResponse,
    HealthResponse,
    JobCreated,
    JobStatus,
)

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_SAFE_FILENAME = re.compile(r"^[a-z0-9_]{1,80}\.docx$")
HOME_PAGE = Path(__file__).parent / "static" / "index.html"

API_GUIDE = """
Multi-agent system (**Planner -> parallel Writers -> Reviewer**) that turns one sentence
into a polished Microsoft Word document.

### Easiest way: the web page
Open **[http://localhost:8000](/)**: type a request, click *Generate*, watch the agent's
task list, then click *Download Word file*.

### Using this page
1. Open **POST /agent** below and click **Try it out**.
2. Replace the example sentence inside `"request": "..."` with the document you want.
3. Click **Execute** and wait 15-40 seconds.
4. In the response, open the **`download_url`** link in a new tab to get the Word file
   (or set `return_file` to `true` to receive the file directly).

Forgot a file name? **GET /documents** lists every generated document with its download link.
"""

# LLM output can contain characters (e.g. U+2011) a Windows cp1252 console can't
# print; without this, log lines containing them are silently dropped.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("agent")


def create_app(settings: Settings | None = None,
               providers: list[ChatProvider] | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        if providers is not None:
            chain, routes = providers, {}
        else:
            by_alias = build_providers(settings)
            chain, routes = list(by_alias.values()), build_routes(settings, by_alias)
        llm = LLMClient(chain, routes=routes, max_retries=settings.llm_max_retries,
                        max_retry_wait=settings.max_retry_wait_seconds)
        app.state.llm = llm
        app.state.agent = DocumentAgent(settings, llm)
        app.state.jobs = JobStore(app.state.agent)
        logger.info("LLM fallback chain: %s", " -> ".join(llm.provider_names))
        for agent, names in llm.describe_routes().items():
            logger.info("  %-8s -> %s", agent, ", ".join(names))
        logger.info("Open http://localhost:8000 (web page) or http://localhost:8000/docs (API)")
        yield
        await app.state.jobs.shutdown()
        await llm.aclose()

    app = FastAPI(title="Autonomous Document Agent", version=__version__,
                  description=API_GUIDE, lifespan=lifespan)

    @app.exception_handler(GuardrailViolation)
    async def _guardrail(_: Request, exc: GuardrailViolation):
        return JSONResponse(status_code=422, content=ErrorResponse(
            error="request_rejected", detail=str(exc)).model_dump())

    @app.exception_handler(LLMUnavailableError)
    async def _llm_down(_: Request, exc: LLMUnavailableError):
        logger.error("LLM unavailable: %s", exc)
        return JSONResponse(status_code=503, content=ErrorResponse(
            error="llm_unavailable",
            detail="No language model could complete the request. Check API keys or try again.",
        ).model_dump())

    def download_url(request: Request, filename: str) -> str:
        return str(request.url_for("download_document", filename=filename))

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    @app.get("/", include_in_schema=False)
    async def home():
        return FileResponse(HOME_PAGE, media_type="text/html")

    @app.get("/health", tags=["system"], response_model=HealthResponse,
             summary="Server status and which model each agent uses")
    async def health(request: Request) -> HealthResponse:
        llm: LLMClient = request.app.state.llm
        return HealthResponse(version=__version__, llm_chain=llm.provider_names,
                              agent_models=llm.describe_routes())

    # ------------------------------------------------------------------ #
    # Main endpoint required by the assignment
    # ------------------------------------------------------------------ #
    @app.post(
        "/agent",
        response_model=AgentResponse,
        tags=["agent"],
        summary="Plan, write, self-review and render a Word document",
        responses={
            200: {"content": {DOCX_MEDIA_TYPE: {}},
                  "description": "JSON report, or the .docx itself when return_file=true"},
            422: {"model": ErrorResponse, "description": "Invalid or rejected request"},
            503: {"model": ErrorResponse, "description": "All LLM providers unavailable"},
        },
    )
    async def run_agent(
        body: AgentRequest,
        request: Request,
        return_file: bool = Query(False, description="Return the .docx file instead of JSON"),
    ):
        result: AgentResponse = await request.app.state.agent.run(body.request, locale=body.locale)
        result.document.download_url = download_url(request, result.document.filename)
        if return_file:
            return FileResponse(settings.output_dir / result.document.filename,
                                media_type=DOCX_MEDIA_TYPE, filename=result.document.filename,
                                headers={"X-Run-Id": result.run_id})
        return result

    # ------------------------------------------------------------------ #
    # Background jobs: same pipeline, but progress can be watched live
    # ------------------------------------------------------------------ #
    @app.post("/jobs", tags=["jobs"], status_code=202, response_model=JobCreated,
              summary="Start the agent in the background and return immediately",
              responses={422: {"model": ErrorResponse, "description": "Rejected request"}})
    async def start_job(body: AgentRequest, request: Request) -> JobCreated:
        validate_request(body.request)  # reject bad input now, not after polling
        job = request.app.state.jobs.start(body.request, body.locale)
        return JobCreated(job_id=job.id,
                          status_url=str(request.url_for("job_status", job_id=job.id)))

    @app.get("/jobs/{job_id}", tags=["jobs"], response_model=JobStatus, name="job_status",
             summary="Live task list of a job, and its result when finished",
             responses={404: {"description": "Unknown job id"}})
    async def job_status(job_id: str, request: Request) -> JobStatus:
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        result = job.result
        if result is not None:
            result = result.model_copy(deep=True)
            result.document.download_url = download_url(request, result.document.filename)
        return JobStatus(job_id=job.id, status=job.status, result=result, error=job.error,
                         todo=[t.model_copy() for t in job.trace.todo])

    # ------------------------------------------------------------------ #
    # Documents
    # ------------------------------------------------------------------ #
    @app.get("/documents", tags=["documents"], response_model=list[DocumentListItem],
             summary="List generated documents, newest first, with download links")
    async def list_documents(
        request: Request, limit: int = Query(20, ge=1, le=200)
    ) -> list[DocumentListItem]:
        files = sorted(settings.output_dir.glob("*.docx"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        return [
            DocumentListItem(
                filename=f.name,
                download_url=download_url(request, f.name),
                size_kb=round(f.stat().st_size / 1024, 1),
                created_at=datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
            )
            for f in files if _SAFE_FILENAME.match(f.name)
        ]

    @app.get("/documents/{filename}", tags=["documents"], name="download_document",
             response_class=FileResponse, summary="Download a generated Word document",
             responses={200: {"content": {DOCX_MEDIA_TYPE: {}},
                              "description": "The .docx file"},
                        404: {"description": "Document not found"}})
    async def download_document(filename: str):
        path = settings.output_dir / filename
        if not _SAFE_FILENAME.match(filename) or not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Document not found. Use GET /documents to see available file names.",
            )
        return FileResponse(path, media_type=DOCX_MEDIA_TYPE, filename=filename)

    return app


app = create_app()
