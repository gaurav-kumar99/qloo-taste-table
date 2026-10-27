"""Qloo API client (search, tags, insights) plus a clearly labeled sample-data stand-in.

Endpoints used (hackathon host, X-Api-Key header):
  GET /search       resolve a name to a Qloo entity id
  GET /v2/tags      resolve a keyword (e.g. "restaurant") to a tag id
  GET /v2/insights  taste-based recommendations from signal.interests.entities,
                    or affinity scores for a given list via filter.results.entities
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

import httpx

BASE_URL = os.getenv("QLOO_API_URL", "https://hackathon.api.qloo.com").rstrip("/")
MAX_SIGNAL = 25

# kind -> (Qloo entity type, tag keyword used to narrow places)
KINDS: dict[str, tuple[str, str | None]] = {
    "restaurant": ("urn:entity:place", "restaurant"),
    "bar": ("urn:entity:place", "bar"),
    "cafe": ("urn:entity:place", "cafe"),
    "movie": ("urn:entity:movie", None),
    "tv_show": ("urn:entity:tv_show", None),
    "artist": ("urn:entity:artist", None),
    "book": ("urn:entity:book", None),
    "brand": ("urn:entity:brand", None),
}


class QlooError(Exception):
    pass


@dataclass
class Entity:
    entity_id: str
    name: str
    kind: str = ""
    affinity: float | None = None
    blurb: str = ""
    image: str | None = None
    lat: float | None = None
    lon: float | None = None
    tags: list[str] = field(default_factory=list)
    address: str = ""
    price_level: int | None = None
    neighborhood: str = ""
    website: str = ""
    hours: dict | None = None
    is_closed: bool = False

    def as_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "kind": self.kind,
            "affinity": self.affinity,
            "blurb": self.blurb,
            "image": self.image,
            "lat": self.lat,
            "lon": self.lon,
            "tags": self.tags,
            "address": self.address,
            "price_level": self.price_level,
            "neighborhood": self.neighborhood,
            "website": self.website,
            "is_closed": self.is_closed,
        }


def _items(data) -> list[dict]:
    """Qloo wraps lists differently per endpoint; accept the shapes we know about."""
    res = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(res, dict):
        res = res.get("entities") or res.get("tags") or res.get("results") or []
    return [x for x in res if isinstance(x, dict)] if isinstance(res, list) else []


def _best_place_tag(ids: list[str]) -> str | None:
    """Prefer the top-level category (urn:tag:category:place:bar) over nested genres like ...:restaurant:bar or ...:hotel:bar."""
    if not ids:
        return None
    return min(ids, key=lambda i: (not i.startswith("urn:tag:category:place:"), i.count(":"), ids.index(i)))


def _image_url(props: dict) -> str | None:
    """Artists/movies use properties.image {url}; places use properties.images [{url}]. Accept both."""
    cands: list = [props.get("image")]
    imgs = props.get("images")
    if isinstance(imgs, list):
        cands.extend(imgs)
    for c in cands:
        url = c.get("url") if isinstance(c, dict) else c
        if isinstance(url, str) and url.startswith("https://"):
            return url
    return None


def _latlon(raw: dict, props: dict) -> tuple[float | None, float | None]:
    """Accept {lat,lon}, {latitude,longitude}, GeoJSON coordinates [lon,lat], or a WKT POINT(lon lat)."""
    loc = raw.get("location") or props.get("location") or {}
    try:
        if isinstance(loc, dict):
            lat = loc.get("lat", loc.get("latitude"))
            lon = loc.get("lon", loc.get("lng", loc.get("longitude")))
            if lat is not None and lon is not None:
                return float(lat), float(lon)
            coords = loc.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return float(coords[1]), float(coords[0])
            wkt = loc.get("wkt")
            if isinstance(wkt, str):
                loc = wkt
        if isinstance(loc, str):
            m = re.search(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)", loc, re.I)
            if m:
                return float(m.group(2)), float(m.group(1))
    except (TypeError, ValueError):
        pass
    return None, None


def _to_entity(raw: dict, kind: str, rank: int, total: int) -> Entity | None:
    eid = raw.get("entity_id") or raw.get("id")
    name = raw.get("name")
    if not eid or not name:
        return None
    aff = (raw.get("query") or {}).get("affinity")
    if aff is None:
        aff = raw.get("affinity")
    props = raw.get("properties") or {}
    blurb = props.get("short_description") or props.get("description") or raw.get("disambiguation") or ""
    if isinstance(blurb, str) and len(blurb) > 160:
        blurb = blurb[:157] + "..."

    # Optional richness. Qloo's shapes vary by entity type, so every lookup is defensive.
    image = _image_url(props)
    lat, lon = _latlon(raw, props)
    tags = [t["name"] for t in (raw.get("tags") or []) if isinstance(t, dict) and t.get("name")][:4]
    address = props.get("address") if isinstance(props.get("address"), str) else ""

    pl = props.get("price_level")
    price_level = int(pl) if isinstance(pl, (int, float)) and not isinstance(pl, bool) and 1 <= pl <= 4 else None
    nb = props.get("neighborhood")
    web = props.get("website")
    hours = props.get("hours") if isinstance(props.get("hours"), dict) else None

    return Entity(str(eid), str(name), kind, float(aff) if aff is not None else None, blurb or "",
                  image=image, lat=lat, lon=lon, tags=tags, address=address,
                  price_level=price_level, neighborhood=nb if isinstance(nb, str) else "",
                  website=web if isinstance(web, str) and web.startswith("https://") else "",
                  hours=hours, is_closed=props.get("is_closed") is True)


class Qloo:
    mode = "live"

    def __init__(self, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"X-Api-Key": api_key},
            timeout=30,
            transport=transport,
        )
        self._tags: dict[str, str | None] = {}
        self.log: list[dict] = []

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _get(self, path: str, params: dict) -> dict:
        r = await self.http.get(path, params=params)
        self.log.append({"call": path, "params": {k: v for k, v in params.items()}, "status": r.status_code})
        if r.status_code >= 400:
            raise QlooError(f"{path} returned {r.status_code}: {r.text[:200]}")
        return r.json()

    async def search(self, name: str, kind: str | None = None) -> Entity | None:
        params: dict = {"query": name, "take": 3}
        if kind and kind in KINDS:
            params["types"] = KINDS[kind][0]
        items = _items(await self._get("/search", params))
        for i, raw in enumerate(items):
            ent = _to_entity(raw, kind or "", i, len(items))
            if ent:
                return ent
        return None

    async def _tag_id(self, kind: str) -> str | None:
        term = KINDS[kind][1]
        if not term:
            return None
        if term in self._tags:
            return self._tags[term]
        tid = None
        try:
            found = [str(t.get("tag_id") or t.get("id")) for t in _items(await self._get("/v2/tags", {"filter.query": term, "take": 10}))
                     if "place" in str(t.get("tag_id") or t.get("id") or "")]
            tid = _best_place_tag(found)
        except QlooError:
            tid = None
        self._tags[term] = tid
        return tid

    async def insights(
        self,
        kind: str,
        signal_ids: list[str],
        city: str | None = None,
        take: int = 20,
        score_ids: list[str] | None = None,
    ) -> list[Entity]:
        """Recommend (or, with score_ids, score a fixed list) from a taste signal."""
        ftype = KINDS[kind][0]
        params: dict = {
            "filter.type": ftype,
            "signal.interests.entities": ",".join(signal_ids[:MAX_SIGNAL]),
            "take": take,
        }
        if ftype == "urn:entity:place" and city:
            params["filter.location.query"] = city
        tag = None if score_ids else await self._tag_id(kind)
        if score_ids:
            params["filter.results.entities"] = ",".join(score_ids)
        elif tag:
            params["filter.tags"] = tag

        items = _items(await self._get("/v2/insights", params))
        if not items and tag:  # tag guess was too narrow; retry untagged
            params.pop("filter.tags", None)
            items = _items(await self._get("/v2/insights", params))

        ents = [e for i, raw in enumerate(items) if (e := _to_entity(raw, kind, i, len(items)))]
        # If Qloo returns no affinity, fall back to rank so downstream scoring still orders results.
        for i, e in enumerate(ents):
            if e.affinity is None:
                e.affinity = round(1.0 - 0.5 * i / max(len(ents), 1), 3)
        return ents


# ---------------------------------------------------------------- sample data

_CATALOG = {
    "restaurant": ["Ember & Oak", "Lotus Noodle House", "Casa Marigold", "The Salted Fig", "Tokyo Alley Izakaya",
                   "Basil & Brine", "Saffron Table", "Little Lisbon", "Grain & Grill", "Pho Corner"],
    "bar": ["The Velvet Room", "Hops & Vinyl", "Rooftop 21", "Copper Door", "Neon Library Bar", "Old Anchor Pub"],
    "cafe": ["Slow Bean", "Paper Moon Cafe", "Third Wave Roasters"],
    "movie": ["Midnight Harbor", "The Last Orchard", "Neon Drift", "Paper Kites", "Static Bloom", "Iron Valley",
              "Soft Landing", "The Quiet Season"],
    "tv_show": ["Dark Meridian", "Small Town Signals", "The Long Table"],
    "artist": ["Luna Vale", "Rook & Rye", "Digital Fern", "The Hollow Pines"],
    "book": ["The Salt Archive", "Glass Cities", "A Field of Echoes"],
    "brand": ["Northwind", "Atlas Goods", "Kiln & Co"],
}


def _h(*parts: str) -> float:
    d = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(d[:4], "big") / 0xFFFFFFFF


class SampleQloo:
    """Deterministic fake data so the app runs without a key. NOT Qloo output."""

    mode = "sample"

    def __init__(self) -> None:
        self.log: list[dict] = []

    async def aclose(self) -> None:
        return None

    async def search(self, name: str, kind: str | None = None) -> Entity | None:
        self.log.append({"call": "sample:/search", "params": {"query": name}, "status": 200})
        return Entity(f"sample:{name.strip().lower()}", name.strip(), kind or "")

    async def insights(self, kind, signal_ids, city=None, take=20, score_ids=None) -> list[Entity]:
        self.log.append({"call": "sample:/v2/insights", "params": {"kind": kind, "signals": len(signal_ids)}, "status": 200})
        names = _CATALOG.get(kind, [])
        pool = names if not score_ids else [n for n in names if f"sample:{n}" in score_ids]
        sig = ",".join(sorted(signal_ids))
        ents = [Entity(f"sample:{n}", n, kind, round(0.25 + 0.7 * _h(sig, n), 3), "Sample data, not a Qloo result") for n in pool]
        if score_ids:  # mimic Qloo dropping some off-taste items from scored lists
            ents = [e for e in ents if e.affinity and e.affinity > 0.32]
        ents.sort(key=lambda e: -(e.affinity or 0))
        return ents[:take]
