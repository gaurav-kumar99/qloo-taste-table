"""Planning agents: a Gemini tool-calling agent grounded in Qloo, with a deterministic fallback."""
from __future__ import annotations

import json
import os
from typing import Callable

from google import genai
from google.genai import types

from .taste import (PLACE_KINDS, Person, alternatives_for, candidates, describe_pick, fairness, is_open_at,
                    minutes_from_label, pick_next, plan_greedy, stop_from_row, weekday_of)

# Use Gemini model, falling back to gemini-2.5-flash if not specified
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TURNS = 10
KIND_ENUM = ["restaurant", "bar", "cafe", "movie", "tv_show", "artist", "book", "brand"]

SYSTEM = """You plan a multi-stop group outing (for example dinner, a movie, then a bar) using Qloo taste data.

Rules:
- Every stop MUST come from a tool result. Never invent venues or titles; use the exact entity_id and name returned.
- Call recommend_for_group once per stop kind. Each row has `fit`, the 0-1 Qloo affinity of each person for that pick.
- Fairness matters more than the single highest average: rotate whose taste leads across stops, and avoid a pick where
  anyone's fit is very low unless nothing better exists. Use recommend_for_person to look for a tie-breaker.
- Do not reuse the same entity twice. Respect the occasion.
- If `feedback` is present, treat it as the group's constraints or mood and choose accordingly. Rows carry
  `price_level` (1 = cheapest to 4 = priciest, null when unknown), `neighborhood` and `tags`; use them for budget or
  vibe requests. Venues known to be closed at the stop time are already removed. Qloo has no dietary data, so when
  feedback depends on something the data cannot verify (for example a dietary need, or a price that is null), say so
  plainly in `fairness_note` instead of claiming it was met.
- Stops listed in `locked` (kind -> entity_id) must be used exactly as given. Never use any id in `excluded`.
- Finish with ONLY a JSON object, no prose, shaped like:
  {"title": str, "summary": str, "stops": [{"kind": str, "time": str, "entity_id": str, "name": str,
   "why": str, "favors": [person names]}], "fairness_note": str}
- Each `why` is one sentence that cites at least two people's fit numbers from the tool data."""

# Gemini Tool Declarations format
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="recommend_for_group",
                description="Qloo picks for the whole group for one stop kind, with each person's affinity for every pick.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "kind": types.Schema(
                            type=types.Type.STRING,
                            enum=KIND_ENUM,
                            description="The category of venue or media."
                        )
                    },
                    required=["kind"],
                ),
            ),
            types.FunctionDeclaration(
                name="recommend_for_person",
                description="Qloo picks driven only by one person's favorites, for tie-breaking when someone is underserved.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "person": types.Schema(
                            type=types.Type.STRING,
                            description="Name of the person."
                        ),
                        "kind": types.Schema(
                            type=types.Type.STRING,
                            enum=KIND_ENUM,
                            description="The category of venue or media."
                        )
                    },
                    required=["person", "kind"],
                ),
            ),
        ]
    )
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
        date: str = "",
    ):
        self.date = date
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
            self.cache[kind] = self._drop_closed(kind, await candidates(self.qloo, self.people, kind, self.city))
            n = len(self.cache[kind])
            self.log(f"Scored {n} {kind.replace('_', ' ')} candidates against each person's own taste")
        return self.cache[kind]

    def _drop_closed(self, kind: str, rows: list[dict]) -> list[dict]:
        """Remove venues Qloo lists as closed, or closed at this stop's time on the chosen date. Unknown hours are kept."""
        if kind not in PLACE_KINDS:
            return rows
        day = weekday_of(self.date)
        minute = minutes_from_label(dict(self.stops).get(kind, ""))
        keep = [r for r in rows if not r.get("is_closed") and is_open_at(r.get("hours"), day, minute) is not False]
        dropped = len(rows) - len(keep)
        if not keep:  # never strand the plan: if everything looks closed, trust the original list
            return rows
        if dropped:
            when = f" at {dict(self.stops)[kind]} on {day.title()}" if day and minute is not None else ""
            self.log(f"Skipped {dropped} {kind} venues that are closed{when}")
        return keep

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
                self.log(f"Gemini is reviewing the top {len(rows)} {kind.replace('_', ' ')} candidates")
                keys = ("entity_id", "name", "blurb", "fit", "mean", "min", "price_level", "neighborhood", "tags")
                return json.dumps([{k: r.get(k) for k in keys} for r in rows])
            if name == "recommend_for_person":
                p = next((x for x in self.people if x.name.lower() == str(args.get("person", "")).lower()), None)
                if not p:
                    return json.dumps({"error": "unknown person"})
                self.log(f"Gemini wants a tie-breaker: {p.name}'s own {kind.replace('_', ' ')} picks")
                ents = await self.qloo.insights(kind, p.entity_ids, self.city, take=6)
                return json.dumps([{"entity_id": e.entity_id, "name": e.name, "affinity": e.affinity} for e in ents])
        except Exception as exc:  # surface tool errors to the model instead of crashing the run
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"unknown tool {name}"})

    # ------------------------------------------------------------ agents
    async def plan(self) -> tuple[str, dict]:
        await self.prefetch()
        agent, result = "fairness-planner", None
        
        # Check for Gemini API key instead of Anthropic
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                self.log("Gemini is weighing the tradeoffs and balancing whose taste leads")
                result = await self._gemini()
                if result:
                    agent = "gemini"
            except Exception as exc:
                self.log(f"Gemini agent failed ({type(exc).__name__}); used the fairness planner instead")
        
        if result is None:
            if self.feedback:
                self.log("Refinement notes are only applied by the AI agent (set GEMINI_API_KEY)")
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

    async def _gemini(self) -> dict | None:
        client = genai.Client()  # Picks up GEMINI_API_KEY automatically
        
        brief = {
            "city": self.city,
            "occasion": self.occasion,
            "feedback": self.feedback,
            "locked": self.locked,
            "excluded": sorted(self.excluded),
            "stops_wanted": [{"kind": k, "time": t} for k, t in self.stops],
            "people": [{"name": p.name, "favorites_resolved": [e.name for e in p.resolved]} for p in self.people],
        }

        chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                tools=TOOLS,
                temperature=0.7,
            )
        )

        # Initial message sending the briefing state
        response = chat.send_message(json.dumps(brief))

        for _ in range(MAX_TURNS):
            # Check if Gemini wants to call functions
            if response.function_calls:
                function_responses = []
                for call in response.function_calls:
                    tool_output = await self.run_tool(call.name, call.args)
                    
                    # Parse JSON output back for Gemini tool response structure
                    try:
                        parsed_output = json.loads(tool_output)
                    except json.JSONDecodeError:
                        parsed_output = {"result": tool_output}

                    function_responses.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": parsed_output}
                        )
                    )
                
                # Send the tool execution results back to the chat loop
                response = chat.send_message(function_responses)
            else:
                # Text response received, try parsing final JSON
                text = response.text or ""
                return self._validate(_extract_json(text))

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