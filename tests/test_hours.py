from app.qloo import _best_place_tag, _to_entity
from app.taste import is_open_at, minutes_from_label, weekday_of


def span(o, c, closed=False):
    return [{"opens": o, "closes": c, "closed": closed}]


def test_weekday_and_time_helpers():
    assert weekday_of("2026-10-31") == "saturday"
    assert weekday_of("garbage") is None and weekday_of("") is None
    assert minutes_from_label("7:00 pm") == 19 * 60
    assert minutes_from_label("12:30 am") == 30 and minutes_from_label("12:15 pm") == 12 * 60 + 15
    assert minutes_from_label("evening") is None


def test_open_closed_and_unknown():
    hrs = {"saturday": span("T07:00:00", "T22:00:00")}
    assert is_open_at(hrs, "saturday", 19 * 60) is True
    assert is_open_at(hrs, "saturday", 23 * 60 + 30) is False
    assert is_open_at(hrs, "saturday", 6 * 60) is False
    assert is_open_at(None, "saturday", 600) is None          # no data: never guess closed
    assert is_open_at(hrs, "sunday", 600) is None             # day missing: unknown
    assert is_open_at({"saturday": []}, "saturday", 600) is None
    assert is_open_at(hrs, None, 600) is None and is_open_at(hrs, "saturday", None) is None
    assert is_open_at({"saturday": span("T07:00:00", "T22:00:00", closed=True)}, "saturday", 19 * 60) is False


def test_late_night_and_24h():
    bar = {"friday": span("T17:00:00", "T02:00:00"), "saturday": span("T17:00:00", "T02:00:00")}
    assert is_open_at(bar, "saturday", 23 * 60 + 30) is True   # open past midnight
    assert is_open_at(bar, "saturday", 1 * 60) is True         # still open from Friday's shift
    assert is_open_at(bar, "saturday", 3 * 60) is False
    assert is_open_at({"saturday": span("T00:00:00", "T00:00:00")}, "saturday", 4 * 60) is True  # 24h


def test_new_entity_fields_parse():
    e = _to_entity({"name": "Bookmarc", "entity_id": "1", "properties": {
        "price_level": 2, "neighborhood": "West Village", "website": "https://example.com",
        "hours": {"saturday": span("T10:00:00", "T20:00:00")}, "is_closed": False}}, "restaurant", 0, 1)
    assert e.price_level == 2 and e.neighborhood == "West Village" and e.website == "https://example.com"
    assert e.hours["saturday"][0]["opens"] == "T10:00:00" and e.is_closed is False
    bad = _to_entity({"name": "X", "entity_id": "2", "properties": {"price_level": 9, "website": "http://x", "is_closed": True}}, "bar", 0, 1)
    assert bad.price_level is None and bad.website == "" and bad.is_closed is True


def test_tag_choice_prefers_top_level_category():
    bar_ids = ["urn:tag:genre:place:restaurant:bar", "urn:tag:category:place:bar", "urn:tag:genre:place:hotel:bar"]
    assert _best_place_tag(bar_ids) == "urn:tag:category:place:bar"
    only_genres = ["urn:tag:genre:place:restaurant:cafe", "urn:tag:genre:place:restaurant"]
    assert _best_place_tag(only_genres) == "urn:tag:genre:place:restaurant"
    assert _best_place_tag([]) is None
