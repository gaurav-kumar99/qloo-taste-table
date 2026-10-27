from app.qloo import _items, _to_entity


def test_place_shape_from_real_qloo_response():
    raw = {
        "name": "Chelsea Market",
        "entity_id": "168322E2",
        "subtype": "urn:entity:place",
        "properties": {
            "address": "75 9th Ave New York, NY 10011",
            "images": [{"url": "https://images.qloo.com/i/168322E2-420x-outside.jpg", "type": "urn:image:place:unknown"}],
        },
        "location": {"lat": 40.7424, "lon": -74.0061},
        "query": {"affinity": 0.87},
        "tags": [{"name": "Food Hall", "tag_id": "urn:tag:genre:place:food_hall"}],
    }
    e = _to_entity(raw, "restaurant", 0, 1)
    assert e.image == "https://images.qloo.com/i/168322E2-420x-outside.jpg"
    assert (e.lat, e.lon) == (40.7424, -74.0061)
    assert e.affinity == 0.87 and e.address.startswith("75 9th Ave") and e.tags == ["Food Hall"]


def test_artist_image_and_geojson_and_wkt_locations():
    a = _to_entity({"name": "Radiohead", "entity_id": "7", "properties": {"image": {"url": "https://images.qloo.com/i/7.jpg"}}}, "artist", 0, 1)
    assert a.image == "https://images.qloo.com/i/7.jpg"
    g = _to_entity({"name": "G", "entity_id": "1", "location": {"coordinates": [-74.0, 40.7]}}, "bar", 0, 1)
    assert (g.lat, g.lon) == (40.7, -74.0)
    w = _to_entity({"name": "W", "entity_id": "2", "location": {"wkt": "POINT(-74.0 40.7)"}}, "bar", 0, 1)
    assert (w.lat, w.lon) == (40.7, -74.0)


def test_non_https_images_and_bad_locations_are_ignored():
    e = _to_entity({"name": "X", "entity_id": "3", "properties": {"images": [{"url": "http://insecure/x.jpg"}]},
                    "location": {"lat": "nope", "lon": None}}, "bar", 0, 1)
    assert e.image is None and e.lat is None and e.lon is None


def test_items_handles_tags_and_entities_wrappers():
    assert _items({"results": {"tags": [{"id": "t"}]}}) == [{"id": "t"}]
    assert _items({"results": {"entities": [{"entity_id": "e"}]}}) == [{"entity_id": "e"}]
    assert _items({"results": [{"entity_id": "s"}]}) == [{"entity_id": "s"}]
