"""FastAPI entry point.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import re
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app import __version__
from app.config import Settings, get_settings
from app.guardrails import GuardrailViolation
from app.llm import ChatProvider, LLMClient, LLMUnavailableError
from app.llm.factory import build_providers, build_routes
from app.orchestrator import DocumentAgent
from app.schemas import AgentRequest, AgentResponse, ErrorResponse

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_SAFE_FILENAME = re.compile(r"^[a-z0-9_]{1,80}\.docx$")

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
        logger.info("LLM fallback chain: %s", " -> ".join(llm.provider_names))
        for agent, names in llm.describe_routes().items():
            logger.info("  %-8s -> %s", agent, ", ".join(names))
        yield
        await llm.aclose()

    app = FastAPI(
        title="Autonomous Document Agent",
        version=__version__,
        description=(
            "Multi-agent system (Planner -> parallel Writers -> Reviewer) that turns a "
            "natural-language request into a polished Microsoft Word document."
        ),
        lifespan=lifespan,
    )

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

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/docs")

    @app.get("/health", tags=["system"])
    async def health(request: Request):
        llm: LLMClient = request.app.state.llm
        return {"status": "ok", "version": __version__,
                "llm_chain": llm.provider_names, "agent_models": llm.describe_routes()}

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
        result: AgentResponse = await request.app.state.agent.run(body.request)
        result.document.download_url = str(request.url_for(
            "download_document", filename=result.document.filename))
        if return_file:
            return FileResponse(settings.output_dir / result.document.filename,
                                media_type=DOCX_MEDIA_TYPE, filename=result.document.filename,
                                headers={"X-Run-Id": result.run_id})
        return result

    @app.get("/documents/{filename}", tags=["agent"], name="download_document",
             response_class=FileResponse)
    async def download_document(filename: str):
        path = settings.output_dir / filename
        if not _SAFE_FILENAME.match(filename) or not path.is_file():
            raise HTTPException(status_code=404, detail="Document not found")
        return FileResponse(path, media_type=DOCX_MEDIA_TYPE, filename=filename)

    return app


app = create_app()
