"""Background jobs so clients can watch the agent's TODO list update live.

`POST /agent` waits for the whole run (as the assignment specifies). `POST /jobs`
returns immediately with a job id; `GET /jobs/{id}` shows the live task list and,
when finished, the same result `/agent` would return. In-memory is enough for a
single process; in production this would sit on a queue such as Redis/Celery.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field

from app.guardrails import GuardrailViolation
from app.llm import LLMUnavailableError
from app.orchestrator import DocumentAgent
from app.schemas import AgentResponse, ErrorResponse
from app.trace import RunTrace

logger = logging.getLogger("agent")


@dataclass
class Job:
    trace: RunTrace
    status: str = "running"
    result: AgentResponse | None = None
    error: ErrorResponse | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.trace.run_id


class JobStore:
    def __init__(self, agent: DocumentAgent, max_jobs: int = 100) -> None:
        self.agent = agent
        self.max_jobs = max_jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()

    def start(self, request: str) -> Job:
        job = Job(trace=RunTrace())
        job.task = asyncio.create_task(self._run(job, request))
        self._jobs[job.id] = job
        while len(self._jobs) > self.max_jobs:  # forget the oldest finished jobs
            oldest_id, oldest = next(iter(self._jobs.items()))
            if oldest.status == "running":
                break
            self._jobs.pop(oldest_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def _run(self, job: Job, request: str) -> None:
        try:
            job.result = await self.agent.run(request, trace=job.trace)
            job.status = "completed"
        except GuardrailViolation as exc:
            job.error = ErrorResponse(error="request_rejected", detail=str(exc))
            job.status = "failed"
        except LLMUnavailableError:
            job.error = ErrorResponse(
                error="llm_unavailable",
                detail="No language model could complete the request. Try again shortly.",
            )
            job.status = "failed"
        except Exception:  # never leave a client polling forever
            logger.exception("[%s] job crashed", job.id)
            job.error = ErrorResponse(error="internal_error", detail="Unexpected server error.")
            job.status = "failed"

    async def shutdown(self) -> None:
        for job in self._jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()
