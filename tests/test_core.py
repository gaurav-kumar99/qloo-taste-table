import httpx
import pytest

from app.qloo import Qloo, SampleQloo
from app.taste import Person, candidates, fairness, plan_greedy, resolve_people


def test_pick_next_rotates_toward_underserved_person():
    from app.taste import pick_next

    rows = [
        {"entity_id": "a", "name": "A", "fit": {"x": 0.9, "y": 0.2}, "min": 0.2},
        {"entity_id": "b", "name": "B", "fit": {"x": 0.3, "y": 0.8}, "min": 0.3},
        {"entity_id": "c", "name": "C", "fit": {"x": 0.9, "y": 0.25}, "min": 0.25},
    ]
    # x has been served a lot already, y not at all -> expect the y-leaning pick
    assert pick_next(rows, {"x": 1.8, "y": 0.2}, set())["entity_id"] == "b"


@pytest.mark.asyncio
async def test_sample_pipeline_builds_fair_plan():
    q = SampleQloo()
    people = [Person("Ana", ["Parasite", "Radiohead"]), Person("Bo", ["Interstellar"])]
    await resolve_people(q, people)
    assert all(p.resolved for p in people)
    stops = [("restaurant", "7pm"), ("movie", "9pm"), ("bar", "11pm")]
    cands = {k: await candidates(q, people, k, "Anywhere") for k, _ in stops}
    plan = plan_greedy(people, cands, stops)
    assert len(plan) == 3
    assert len({s["entity_id"] for s in plan}) == 3
    assert set(fairness(people, plan)["per_person"]) == {"Ana", "Bo"}


@pytest.mark.asyncio
async def test_live_client_parses_qloo_shapes_and_sends_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["key"] = req.headers.get("x-api-key")
        if req.url.path == "/search":
            return httpx.Response(200, json={"results": [{"entity_id": "E1", "name": "Parasite"}]})
        if req.url.path == "/v2/tags":
            return httpx.Response(200, json={"results": {"tags": [{"id": "urn:tag:category:place:restaurant"}]}})
        return httpx.Response(200, json={"results": {"entities": [
            {"entity_id": "P1", "name": "Place One", "query": {"affinity": 0.91}},
            {"entity_id": "P2", "name": "Place Two"},
        ]}})

    q = Qloo("secret", transport=httpx.MockTransport(handler))
    ent = await q.search("Parasite")
    assert ent and ent.entity_id == "E1" and seen["key"] == "secret"
    ents = await q.insights("restaurant", ["E1"], "Paris")
    assert ents[0].affinity == 0.91 and ents[1].affinity is not None  # rank fallback fills the gap
    await q.aclose()
