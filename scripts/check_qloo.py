"""Compact check of the Qloo fields Taste Table depends on (no key is printed).

PowerShell:
    $env:QLOO_API_KEY = "your-key"
    py -X utf8 scripts/check_qloo.py | Out-File -Encoding utf8 qloo_out2.txt
"""
import json
import os
import sys

import httpx

KEY = os.getenv("QLOO_API_KEY", "").strip()
BASE = os.getenv("QLOO_API_URL", "https://hackathon.api.qloo.com").rstrip("/")
CITY = os.getenv("CHECK_CITY", "New York, NY")
if not KEY:
    sys.exit("Set QLOO_API_KEY first.")

client = httpx.Client(base_url=BASE, headers={"X-Api-Key": KEY}, timeout=30)


def brief(x):
    p = x.get("properties") or {}
    return {
        "name": x.get("name"),
        "entity_id": x.get("entity_id"),
        "query": x.get("query"),
        "location": x.get("location"),
        "popularity": x.get("popularity"),
        "property_keys": sorted(p.keys()),
        "image": p.get("image"),
        "images_first": (p.get("images") or [None])[0],
        "price_level": p.get("price_level"),
        "address": p.get("address"),
        "short_description": (p.get("short_description") or p.get("description") or "")[:80],
        "tags": [t.get("name") for t in (x.get("tags") or [])[:5] if isinstance(t, dict)],
    }


def call(title, path, params, full=0):
    print(f"\n=== {title}\nGET {path}")
    print("params:", json.dumps(params)[:300])
    try:
        r = client.get(path, params=params)
    except Exception as exc:
        print("REQUEST FAILED:", type(exc).__name__, exc)
        return []
    print("status:", r.status_code)
    if r.status_code >= 400:
        print("body:", r.text[:400])
        return []
    data = r.json()
    for k in ("warnings", "query"):
        if isinstance(data, dict) and data.get(k):
            print(f"{k}:", json.dumps(data[k])[:400])
    res = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(res, dict):
        res = res.get("entities") or res.get("tags") or []
    res = [x for x in res if isinstance(x, dict)]
    print("count:", len(res))
    for i, x in enumerate(res):
        print(json.dumps(brief(x) if "entity_id" in x else {"id": x.get("id"), "name": x.get("name")}, indent=1)[:1500])
    return res


a = call("Search a movie", "/search", {"query": "Parasite", "take": 1})
b = call("Search an artist", "/search", {"query": "Radiohead", "take": 1})
c = call("Search a restaurant brand", "/search", {"query": "Shake Shack", "take": 1})
ids = [x["entity_id"] for x in a + b + c if x.get("entity_id")]

for term in ("bar", "cafe"):
    call(f"Tag lookup: {term}", "/v2/tags", {"filter.query": term, "take": 5})

if ids:
    places = call("Group restaurants (signal = all favorites)", "/v2/insights", {
        "filter.type": "urn:entity:place", "signal.interests.entities": ",".join(ids),
        "filter.location.query": CITY, "take": 6})
    pids = [x["entity_id"] for x in places if x.get("entity_id")]
    if pids and b:
        call("Per-person scoring (signal = Radiohead only, scored against the restaurants above)", "/v2/insights", {
            "filter.type": "urn:entity:place", "signal.interests.entities": b[0]["entity_id"],
            "filter.location.query": CITY, "filter.results.entities": ",".join(pids), "take": len(pids)})
    call("Group movies", "/v2/insights", {
        "filter.type": "urn:entity:movie", "signal.interests.entities": ",".join(ids), "take": 3})
else:
    print("\nNo entity ids from /search, so insights checks were skipped.")
