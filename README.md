# Taste Table

An agent that plans a multi-stop group evening (dinner, a movie, a bar) from what each person loves, using the [Qloo](https://qloo.com) taste API. Built for the Qloo Agentic Hackathon.

The problem: group plans usually go to the loudest voice or the lowest common denominator. Taste Table asks Qloo how well each *individual* fits each candidate, then rotates whose taste leads across the evening and reports who got what.

## How it works

1. **Resolve.** Each person's favorites (films, artists, restaurants, brands) are matched to Qloo entities with `GET /search`.
2. **Candidates.** For each stop, `GET /v2/insights` with everyone's entities as the signal returns Qloo's picks for the group (places are restricted to the chosen city).
3. **Per-person scoring.** The same insights call is repeated once per person with `filter.results.entities` set to those candidates, giving each person's own affinity for each pick.
4. **Plan.** A Claude tool-calling agent (`recommend_for_group`, `recommend_for_person`) chooses the stops, rotating whose taste leads. Without an Anthropic key, a deterministic fairness planner does the same job by up-weighting whoever has been served least so far.
5. **Guardrail.** Every stop must exist in the Qloo results. Anything the model proposes that does not is replaced with a Qloo-backed pick.
6. **Report.** The UI shows per-person fit bars for every stop and the average fit per person across the evening.

## Features

- **Live agent progress.** The UI streams what the agent is doing as it happens (`POST /api/plan/stream`, newline-delimited JSON).
- **Fairness view.** Per-person affinity bars on every stop, a "leans toward" badge, and average fit per person across the evening.
- **Swap alternatives.** Each stop offers runner-up picks with their per-person fit. Swapping re-computes fairness instantly in the browser.
- **Pin and refine.** Pin stops you like, type a constraint ("Sam is vegetarian, keep it chill"), and the agent re-plans around it. Budget and vibe requests use Qloo's price level and tags. Qloo has no dietary data, so the agent is told to say when it cannot verify a request.
- **Opening hours.** Places that Qloo lists as closed at a stop's time on your chosen date are filtered out; places with no hours data are kept.
- **Rich cards.** Images, tags and addresses are shown when Qloo returns them, plus a route map when it returns coordinates.
- **Neon atmosphere.** A blurred, looping neon background and a glowing cursor trail, both drawn on canvas (no video files). Both are disabled or reduced for visitors who prefer reduced motion.
- **Take it with you.** Copy the plan or download a calendar (.ics) file.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in keys (or export them)
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

On Windows PowerShell, use `py -m pip install -r requirements.txt`, set keys with `$env:QLOO_API_KEY = "..."`, and start with `py -m uvicorn app.main:app --reload`.

| Variable | Purpose |
| --- | --- |
| `QLOO_API_KEY` | Hackathon key from the Qloo developer guide form. If empty, the app runs in a clearly labeled **sample data** mode (fake, deterministic results, not Qloo output). |
| `QLOO_API_URL` | Defaults to `https://hackathon.api.qloo.com`. |
| `ANTHROPIC_API_KEY` | Enables the Claude agent. If empty, the fairness planner is used. |
| `ANTHROPIC_MODEL` | Defaults to `claude-sonnet-5-5`. |

## Tests

```bash
pytest -q
```

Covers the fairness rotation, the end-to-end sample pipeline, and the live client's request/response handling against a mocked Qloo.

## Layout

- `app/qloo.py` Qloo client (search, tags, insights) and the sample-data stand-in
- `app/taste.py` group candidates, per-person scoring, fairness planner
- `app/agent.py` Claude tool-calling agent, grounding guardrail
- `app/main.py` FastAPI app (`POST /api/plan`, `GET /api/health`)
- `static/index.html` single-page UI

## License

MIT. Not affiliated with or endorsed by Qloo.
