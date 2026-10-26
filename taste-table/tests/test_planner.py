import json

import pytest

from app.agent import Planner, finish
from app.qloo import SampleQloo
from app.taste import Person, alternatives_for, candidates, plan_greedy, resolve_people

STOPS = [("restaurant", "7:00 pm"), ("movie", "9:15 pm"), ("bar", "11:30 pm")]


async def _setup():
    q = SampleQloo()
    people = [Person("Maya", ["Parasite", "Radiohead"]), Person("Dev", ["Interstellar"]), Person("Sam", ["Spirited Away"])]
    await resolve_people(q, people)
    cands = {k: await candidates(q, people, k, "Anywhere") for k, _ in STOPS}
    return q, people, cands


@pytest.mark.asyncio
async def test_locked_and_excluded_are_honored():
    _, people, cands = await _setup()
    base = plan_greedy(people, cands, STOPS)
    pinned = base[1]["entity_id"]
    banned = base[0]["entity_id"]
    plan = plan_greedy(people, cands, STOPS, locked={"movie": pinned}, excluded={banned})
    assert plan[1]["entity_id"] == pinned
    assert banned not in {s["entity_id"] for s in plan}


@pytest.mark.asyncio
async def test_alternatives_are_unchosen_and_carry_fit():
    _, people, cands = await _setup()
    plan = plan_greedy(people, cands, STOPS)
    alternatives_for(plan, cands)
    chosen = {s["entity_id"] for s in plan}
    for s in plan:
        assert len(s["alternatives"]) <= 3
        for a in s["alternatives"]:
            assert a["entity_id"] not in chosen
            assert set(a["fit"]) == {"Maya", "Dev", "Sam"}


@pytest.mark.asyncio
async def test_planner_without_anthropic_key_uses_fairness_planner(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    q, people, _ = await _setup()
    events = []
    planner = Planner(q, people, "Anywhere", "Friday", STOPS, feedback="keep it chill", on_event=events.append)
    agent, result = await planner.plan()
    result = finish(people, result)
    assert agent == "fairness-planner"
    assert len(result["stops"]) == 3
    assert result["fairness"]["gap"] >= 0
    assert events, "progress events should be emitted for the live UI"
    assert any("Claude agent" in e for e in events)  # feedback needs the Claude agent; user is told so


@pytest.mark.asyncio
async def test_validate_replaces_invented_stops_and_keeps_locks(monkeypatch):
    q, people, _ = await _setup()
    planner = Planner(q, people, "Anywhere", "Friday", STOPS)
    await planner.prefetch()
    real_movie = planner.cache["movie"][0]["entity_id"]
    planner.locked = {"movie": real_movie}
    proposal = {"title": "t", "summary": "s", "fairness_note": "", "stops": [
        {"kind": "restaurant", "entity_id": "made-up", "name": "Totally Invented Bistro", "why": "x"},
        {"kind": "movie", "entity_id": "something-else", "name": "Other", "why": "y"},
        {"kind": "bar", "entity_id": "nope", "name": "Imaginary Bar", "why": "z"},
    ]}
    out = planner._validate(proposal)
    names = [s["name"] for s in out["stops"]]
    assert "Totally Invented Bistro" not in names and "Imaginary Bar" not in names
    assert out["stops"][1]["entity_id"] == real_movie


def test_stream_endpoint_emits_steps_then_result(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.delenv("QLOO_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = {"city": "NYC", "people": [{"name": "A", "favorites": ["Parasite"]}, {"name": "B", "favorites": ["Interstellar"]}]}
    with TestClient(app) as c:
        r = c.post("/api/plan/stream", json=body)
    assert r.status_code == 200
    msgs = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    kinds = [m["type"] for m in msgs]
    assert "step" in kinds and kinds[-1] == "result"
    assert msgs[-1]["data"]["mode"] == "sample"
    assert len(msgs[-1]["data"]["plan"]["stops"]) == 3
