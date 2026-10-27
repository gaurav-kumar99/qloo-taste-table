"""Group taste logic on top of Qloo: resolve favorites, build candidates, score per person."""
from __future__ import annotations

import asyncio
import datetime
import re
from dataclasses import dataclass, field

from .qloo import Entity, QlooError

DEFAULT_STOPS = [("restaurant", "7:00 pm"), ("movie", "9:15 pm"), ("bar", "11:30 pm")]
PLACE_KINDS = {"restaurant", "bar", "cafe"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def weekday_of(date: str) -> str | None:
    try:
        return DAYS[datetime.date.fromisoformat(date).weekday()]
    except (ValueError, TypeError):
        return None


def minutes_from_label(label: str) -> int | None:
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*(am|pm)", label or "", re.I)
    if not m:
        return None
    return (int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)) * 60 + int(m.group(2))


def _clock(v) -> int | None:
    m = re.match(r"T?(\d{1,2}):(\d{2})", v) if isinstance(v, str) else None
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def is_open_at(hours, weekday: str | None, minute: int | None) -> bool | None:
    """True/False when a venue's listed hours say so; None when the data can't tell (never guess closed)."""
    if not isinstance(hours, dict) or weekday not in DAYS or minute is None:
        return None
    today = hours.get(weekday)
    if not isinstance(today, list) or not today:
        return None
    for s in today:
        if not isinstance(s, dict) or s.get("closed"):
            continue
        o, c = _clock(s.get("opens")), _clock(s.get("closes"))
        if o is None or c is None:
            return None
        if c <= o:  # runs past midnight (or open 24h)
            if minute >= o:
                return True
        elif o <= minute < c:
            return True
    prev = hours.get(DAYS[(DAYS.index(weekday) - 1) % 7])  # yesterday's late-night spill into today
    for s in prev if isinstance(prev, list) else []:
        if isinstance(s, dict) and not s.get("closed"):
            o, c = _clock(s.get("opens")), _clock(s.get("closes"))
            if o is not None and c is not None and c < o and minute < c:
                return True
    return False


@dataclass
class Person:
    name: str
    favorites: list[str]
    resolved: list[Entity] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def entity_ids(self) -> list[str]:
        return [e.entity_id for e in self.resolved]


async def _safe_insights(qloo, kind: str, entity_ids: list[str], city: str | None, take: int = 12, score_ids: list[str] | None = None, retries: int = 3, backoff: float = 1.5):
    """Wrapper around qloo.insights with automatic retry on rate limit (429) errors."""
    for attempt in range(retries):
        try:
            if score_ids:
                return await qloo.insights(kind, entity_ids, city, take=take, score_ids=score_ids)
            else:
                return await qloo.insights(kind, entity_ids, city, take=take)
        except QlooError as e:
            # Check if it's a rate limit / 429 error message
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < retries - 1:
                    await asyncio.sleep(backoff ** (attempt + 1))
                    continue
            raise e
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < retries - 1:
                    await asyncio.sleep(backoff ** (attempt + 1))
                    continue
            raise e
    return []


async def resolve_people(qloo, people: list[Person]) -> None:
    async def one(p: Person, fav: str):
        for attempt in range(3):
            try:
                return p, fav, await qloo.search(fav)
            except QlooError as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < 2:
                        await asyncio.sleep(1.5 ** (attempt + 1))
                        continue
                return p, fav, None
            except Exception:
                return p, fav, None
        return p, fav, None

    jobs = [one(p, f) for p in people for f in p.favorites if f.strip()]
    for p, fav, ent in await asyncio.gather(*jobs):
        (p.resolved.append(ent) if ent else p.unresolved.append(fav))


async def candidates(qloo, people: list[Person], kind: str, city: str | None, take: int = 12) -> list[dict]:
    """Qloo picks for the whole group, then each person's own affinity for those exact picks with rate-limit protection."""
    group_ids = list(dict.fromkeys(i for p in people for i in p.entity_ids))
    if not group_ids:
        return []
    
    base = await _safe_insights(qloo, kind, group_ids, city, take)
    if not base:
        return []
    
    ids = [e.entity_id for e in base]
    
    # Fetch per-person insights safely with staggered execution to avoid hitting rate caps
    per_person = []
    for p in people:
        if p.entity_ids:
            res = await _safe_insights(qloo, kind, p.entity_ids, city, take=len(ids), score_ids=ids)
            per_person.append(res)
        else:
            per_person.append([])

    rows = []
    for e in base:
        fit = {}
        for p, scored in zip(people, per_person):
            hit = next((s for s in scored if s.entity_id == e.entity_id), None)
            # Qloo omits off-taste items from a scored list, so a miss means weak fit, not unknown.
            fit[p.name] = round(hit.affinity, 3) if hit and hit.affinity is not None else 0.0
        vals = list(fit.values())
        rows.append({
            "entity_id": e.entity_id,
            "name": e.name,
            "kind": kind,
            "blurb": e.blurb,
            "image": e.image,
            "lat": e.lat,
            "lon": e.lon,
            "tags": e.tags,
            "address": e.address,
            "price_level": e.price_level,
            "neighborhood": e.neighborhood,
            "website": e.website,
            "hours": e.hours,
            "is_closed": e.is_closed,
            "group_affinity": e.affinity,
            "fit": fit,
            "mean": round(sum(vals) / len(vals), 3),
            "min": round(min(vals), 3),
        })
    rows.sort(key=lambda r: -(r["mean"] + 0.3 * r["min"]))
    return rows


async def _none():
    return []


def _score(row: dict, weights: dict[str, float]) -> float:
    tw = sum(weights.values()) or 1.0
    weighted = sum(row["fit"].get(n, 0.0) * w for n, w in weights.items()) / tw
    return weighted + 0.3 * row["min"]


def pick_next(rows: list[dict], totals: dict[str, float], chosen: set[str]) -> dict | None:
    """Pick the best remaining row, up-weighting people whose taste has been served least so far."""
    top = max(totals.values()) if any(totals.values()) else 0.0
    weights = {n: 1.0 + 2.0 * (top - t) for n, t in totals.items()}
    pool = [r for r in rows if r["entity_id"] not in chosen]
    return max(pool, key=lambda r: _score(r, weights)) if pool else None


def describe_pick(row: dict) -> tuple[str, list[str]]:
    fit = row["fit"]
    best = max(fit, key=fit.get)
    worst = min(fit, key=fit.get)
    favors = [best]
    if len(fit) == 1:
        return f"Strong match for {best} ({fit[best]:.2f}).", favors
    why = f"Best fit for {best} ({fit[best]:.2f}); {worst} still gets {fit[worst]:.2f}."
    return why, favors


def stop_from_row(row: dict, kind: str, time: str, why: str, favors: list[str]) -> dict:
    return {"kind": kind, "time": time, "name": row["name"], "entity_id": row["entity_id"],
            "why": why, "favors": favors, "fit": row["fit"], "blurb": row.get("blurb", ""),
            "image": row.get("image"), "lat": row.get("lat"), "lon": row.get("lon"),
            "tags": row.get("tags", []), "address": row.get("address", ""),
            "price_level": row.get("price_level"), "neighborhood": row.get("neighborhood", ""),
            "website": row.get("website", "")}


def plan_greedy(people: list[Person], cands: dict[str, list[dict]], stops: list[tuple[str, str]],
                locked: dict[str, str] | None = None, excluded: set[str] | None = None) -> list[dict]:
    """Fairness-first planner. `locked` pins a kind to an entity id; `excluded` ids are never picked."""
    locked, excluded = locked or {}, excluded or set()
    totals = {p.name: 0.0 for p in people}
    chosen: set[str] = set(excluded)
    out = []
    for kind, time in stops:
        rows = cands.get(kind, [])
        row = next((r for r in rows if r["entity_id"] == locked.get(kind)), None) or pick_next(rows, totals, chosen)
        if not row:
            continue
        chosen.add(row["entity_id"])
        for n in totals:
            totals[n] += row["fit"].get(n, 0.0)
        why, favors = describe_pick(row)
        out.append(stop_from_row(row, kind, time, why, favors))
    return out


def alternatives_for(stops: list[dict], cands: dict[str, list[dict]], limit: int = 3) -> None:
    """Attach the best unchosen runner-ups to each stop so the UI can swap them in instantly."""
    used = {s["entity_id"] for s in stops}
    for s in stops:
        pool = [r for r in cands.get(s["kind"], []) if r["entity_id"] not in used]
        pool.sort(key=lambda r: -(r["mean"] + 0.3 * r["min"]))
        s["alternatives"] = [
            {k: r.get(k) for k in ("entity_id", "name", "fit", "blurb", "image", "lat", "lon", "tags", "address",
                                   "price_level", "neighborhood", "website")}
            for r in pool[:limit]
        ]


def fairness(people: list[Person], stops: list[dict]) -> dict:
    """Average fit per person across the evening, and the gap between best and worst served."""
    if not stops:
        return {"per_person": {}, "gap": 0.0, "verdict": "No stops planned."}
    per = {p.name: round(sum(s["fit"].get(p.name, 0.0) for s in stops) / len(stops), 3) for p in people}
    gap = round(max(per.values()) - min(per.values()), 3)
    # Qloo affinities for real picks sit in a tight band, so a small gap is already meaningful.
    best = max(per, key=per.get)
    verdict = "Balanced evening." if gap < 0.04 else f"Leans toward {best}'s taste. Swap a stop to rebalance."
    return {"per_person": per, "gap": gap, "verdict": verdict}
