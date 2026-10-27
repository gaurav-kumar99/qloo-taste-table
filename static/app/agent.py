"""Planning agents: a Claude tool-calling agent grounded in Qloo, with a deterministic fallback."""
from __future__ import annotations

import json
import os
from typing import Callable

from .taste import Person, alternatives_for, candidates, describe_pick, fairness, pick_next, plan_greedy, stop_from_row

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 10
KIND_ENUM = ["restaurant", "bar", "cafe", "movie", "tv_show", "artist", "book", "brand"]

SYSTEM = """You plan a multi-stop group outing (for example dinner, a movie, then a bar) using Qloo taste data.

Rules:
- Every stop MUST come from a tool result. Never invent venues or titles; use the exact entity_id and name returned.
- Call recommend_for_group once per stop kind. Each row has `fit`, the 0-1 Qloo affinity of each person for that pick.
- Fairness matters more than the single highest average: rotate whose taste leads across stops, and avoid a pick where
  anyone's fit is very low unless nothing better exists. Use recommend_for_person to look for a tie-breaker.
- Do not reuse the same entity twice. Respect the occasion.
- If `feedback` is present, treat it as the group's constraints or mood and choose accordingly. Qloo data has no price,
  dietary or opening-hours information, so when feedback depends on something the data cannot verify, say so plainly in
  `fairness_note` instead of claiming it was met.
- Stops listed in `locked` (kind -> entity_id) must be used exactly as given. Never use any id in `excluded`.
- Finish with ONLY a JSON object, no prose, shaped like:
  {"title": str, "summary": str, "stops": [{"kind": str, "time": str, "entity_id": str, "name": str,
   "why": str, "favors": [person names]}], "fairness_note": str}
- Each `why` is one sentence that cites at least two people's fit numbers from the tool data."""

TOOLS = [
    {
        "name": "recommend_for_group",
        "description": "Qloo picks for the whole group for one stop kind, with each person's affinity for every pick.",
        "input_schema": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": KIND_ENUM}},
            "required": ["kind"],
        },
    },
    {
        "name": "recommend_for_person",
        "description": "Qloo picks driven only by one person's favorites, for tie-breaking when someone is underserved.",
        "input_schema": {
            "type": "object",
            "properties": {"person": {"type": "string"}, "kind": {"type": "string", "enum": KIND_ENUM}},
            "required": ["person", "kind"],
        },
    },
]


class Planner:
    def __init__(
        self,
        qloo,
        people: list[Person],
        city: str | None,
        occasion: str,
        stops: list[tuple[str, str]],
        feedback: str = "",
        locked: dict[str, str] | None = None,
        excluded: set[str] | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        self.qloo, self.people, self.city, self.occasion, self.stops = qloo, people, city, occasion, stops
        self.feedback, self.locked, self.excluded = feedback, locked or {}, excluded or set()
        self.on_event = on_event
        self.cache: dict[str, list[dict]] = {}
        self.trace: list[str] = []

    def log(self, msg: str) -> None:
        self.trace.append(msg)
        if self.on_event:
            self.on_event(msg)

    async def group_rows(self, kind: str) -> list[dict]:
        if kind not in self.cache:
            self.log(f"Asking Qloo what this group loves in {kind.replace('_', ' ')}s...")
            self.cache[kind] = await candidates(self.qloo, self.people, kind, self.city)
            n = len(self.cache[kind])
            self.log(f"Scored {n} {kind.replace('_', ' ')} candidates against each person's own taste")
        return self.cache[kind]

    async def prefetch(self) -> None:
        for kind, _ in self.stops:
            await self.group_rows(kind)

    # ------------------------------------------------------------ tools
    async def run_tool(self, name: str, args: dict) -> str:
        kind = args.get("kind", "")
        try:
            if kind not in KIND_ENUM:
                return json.dumps({"error": f"unknown kind {kind!r}"})
            if name == "recommend_for_group":
                rows = (await self.group_rows(kind))[:8]
                self.log(f"Claude is reviewing the top {len(rows)} {kind.replace('_', ' ')} candidates")
                return json.dumps([{k: r[k] for k in ("entity_id", "name", "blurb", "fit", "mean", "min")} for r in rows])
            if name == "recommend_for_person":
                p = next((x for x in self.people if x.name.lower() == str(args.get("person", "")).lower()), None)
                if not p:
                    return json.dumps({"error": "unknown person"})
                self.log(f"Claude wants a tie-breaker: {p.name}'s own {kind.replace('_', ' ')} picks")
                ents = await self.qloo.insights(kind, p.entity_ids, self.city, take=6)
                return json.dumps([{"entity_id": e.entity_id, "name": e.name, "affinity": e.affinity} for e in ents])
        except Exception as exc:  # surface tool errors to the model instead of crashing the run
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"unknown tool {name}"})

    # ------------------------------------------------------------ agents
    async def plan(self) -> tuple[str, dict]:
        await self.prefetch()
        agent, result = "fairness-planner", None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.log("Claude is weighing the tradeoffs and balancing whose taste leads")
                result = await self._claude()
                if result:
                    agent = "claude"
            except Exception as exc:
                self.log(f"Claude agent failed ({type(exc).__name__}); used the fairness planner instead")
        if result is None:
            if self.feedback:
                self.log("Refinement notes are only applied by the Claude agent (set ANTHROPIC_API_KEY)")
            self.log("Balancing the evening so nobody's taste is left out")
            greedy = plan_greedy(self.people, self.cache, self.stops, self.locked, self.excluded)
            result = {
                "title": "Your evening, balanced by taste",
                "summary": f"A {len(greedy)}-stop plan chosen so everyone's taste is served.",
                "stops": greedy,
                "fairness_note": "",
            }
        alternatives_for(result["stops"], self.cache)
        return agent, result

    async def _claude(self) -> dict | None:
        import anthropic

        client = anthropic.AsyncAnthropic()
        brief = {
            "city": self.city,
            "occasion": self.occasion,
            "feedback": self.feedback,
            "locked": self.locked,
            "excluded": sorted(self.excluded),
            "stops_wanted": [{"kind": k, "time": t} for k, t in self.stops],
            "people": [{"name": p.name, "favorites_resolved": [e.name for e in p.resolved]} for p in self.people],
        }
        messages = [{"role": "user", "content": json.dumps(brief)}]
        for _ in range(MAX_TURNS):
            resp = await client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM, tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                text = "".join(b.text for b in resp.content if b.type == "text")
                return self._validate(_extract_json(text))
            results = []
            for c in calls:
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": await self.run_tool(c.name, c.input)})
            messages.append({"role": "user", "content": results})
        return None

    def _validate(self, plan: dict | None) -> dict | None:
        """Keep only stops that exist in Qloo results; honor locks and exclusions; repair the rest."""
        if not plan or not isinstance(plan.get("stops"), list):
            return None
        totals = {p.name: 0.0 for p in self.people}
        chosen: set[str] = set(self.excluded)
        stops = []
        for i, (kind, time) in enumerate(self.stops):
            rows = self.cache.get(kind, [])
            proposed = plan["stops"][i] if i < len(plan["stops"]) and isinstance(plan["stops"][i], dict) else {}
            row = next((r for r in rows if r["entity_id"] == self.locked.get(kind)), None)
            why, favors = proposed.get("why", ""), proposed.get("favors", [])
            if row is not None:
                if proposed.get("entity_id") != row["entity_id"]:
                    why, favors = describe_pick(row)
            else:
                row = next((r for r in rows if r["entity_id"] == proposed.get("entity_id") and r["entity_id"] not in chosen), None) \
                    or next((r for r in rows if r["name"].lower() == str(proposed.get("name", "")).lower() and r["entity_id"] not in chosen), None)
                if row is None:
                    row = pick_next(rows, totals, chosen)
                    if row is None:
                        continue
                    why, favors = describe_pick(row)
                    self.log(f"Replaced an ungrounded {kind} suggestion with a Qloo-backed pick")
            chosen.add(row["entity_id"])
            for n in totals:
                totals[n] += row["fit"].get(n, 0.0)
            if not isinstance(favors, list):
                favors = []
            stops.append(stop_from_row(row, kind, proposed.get("time") or time, str(why), favors))
        if not stops:
            return None
        plan["stops"] = stops
        return plan


def _extract_json(text: str) -> dict | None:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a : b + 1])
    except json.JSONDecodeError:
        return None


def finish(people: list[Person], plan: dict) -> dict:
    plan["fairness"] = fairness(people, plan["stops"])
    if not plan.get("fairness_note"):
        plan["fairness_note"] = plan["fairness"]["verdict"]
    return plan
