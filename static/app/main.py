from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import Planner, finish
from .qloo import KINDS, Qloo, QlooError, SampleQloo
from .taste import DEFAULT_STOPS, Person, resolve_people

app = FastAPI(title="Taste Table")
STATIC = Path(__file__).resolve().parent.parent / "static"
TIMES = {"cafe": "4:00 pm", "restaurant": "7:00 pm", "movie": "9:15 pm", "bar": "11:30 pm"}


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    favorites: list[str] = Field(min_length=1, max_length=12)


class PlanIn(BaseModel):
    city: str = Field(default="", max_length=80)
    occasion: str = Field(default="A night out", max_length=200)
    people: list[PersonIn] = Field(min_length=1, max_length=6)
    stops: list[str] = Field(default_factory=lambda: [k for k, _ in DEFAULT_STOPS])
    feedback: str = Field(default="", max_length=300)  # refinement: "Sam is vegetarian, keep it chill"
    locked: dict[str, str] = Field(default_factory=dict)  # kind -> entity_id to keep
    excluded: list[str] = Field(default_factory=list, max_length=50)  # entity ids to avoid


def make_qloo():
    key = os.getenv("QLOO_API_KEY", "").strip()
    return Qloo(key) if key else SampleQloo()


async def run_plan(body: PlanIn, emit: Callable[[str], None] | None = None) -> dict:
    stops = [(k, TIMES.get(k, "evening")) for k in body.stops if k in KINDS]
    if not stops:
        raise HTTPException(400, "No valid stop kinds")
    names = [p.name.strip() for p in body.people]
    if len(set(n.lower() for n in names)) != len(names):
        raise HTTPException(400, "Each person needs a unique name")

    people = [Person(p.name.strip(), [f.strip() for f in p.favorites if f.strip()]) for p in body.people]
    qloo = make_qloo()
    try:
        if emit:
            emit("Matching everyone's favorites to Qloo's taste graph...")
        await resolve_people(qloo, people)
        if not any(p.resolved for p in people):
            raise HTTPException(422, "Qloo could not match any of the favorites. Try more widely known names.")
        if emit:
            matched = sum(len(p.resolved) for p in people)
            emit(f"Matched {matched} favorites across {len(people)} people")
        planner = Planner(
            qloo, people, body.city.strip() or None, body.occasion.strip(), stops,
            feedback=body.feedback.strip(), locked=body.locked, excluded=set(body.excluded), on_event=emit,
        )
        agent, result = await planner.plan()
        result = finish(people, result)
        return {
            "mode": qloo.mode,
            "agent": agent,
            "plan": result,
            "people": [{"name": p.name, "matched": [e.name for e in p.resolved], "unmatched": p.unresolved} for p in people],
            "trace": planner.trace + [f"{c['call']} -> {c['status']}" for c in qloo.log],
        }
    except QlooError as exc:
        raise HTTPException(502, f"Qloo API error: {exc}")
    finally:
        await qloo.aclose()


@app.get("/api/health")
def health():
    return {"ok": True, "qloo": "live" if os.getenv("QLOO_API_KEY") else "sample",
            "agent": "claude" if os.getenv("ANTHROPIC_API_KEY") else "fairness-planner"}


@app.post("/api/plan")
async def plan(body: PlanIn):
    return await run_plan(body)


@app.post("/api/plan/stream")
async def plan_stream(body: PlanIn):
    """Newline-delimited JSON: {"type":"step"|"result"|"error", ...} so the UI can show the agent working."""
    q: asyncio.Queue = asyncio.Queue()

    async def worker():
        try:
            data = await run_plan(body, lambda m: q.put_nowait({"type": "step", "message": m}))
            q.put_nowait({"type": "result", "data": data})
        except HTTPException as exc:
            q.put_nowait({"type": "error", "message": str(exc.detail)})
        except Exception as exc:  # never leave the stream hanging
            q.put_nowait({"type": "error", "message": f"Unexpected error ({type(exc).__name__})"})
        finally:
            q.put_nowait(None)

    async def gen():
        task = asyncio.create_task(worker())
        while (item := await q.get()) is not None:
            yield json.dumps(item) + "\n"
        await task

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
