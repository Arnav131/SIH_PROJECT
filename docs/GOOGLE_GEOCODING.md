# Reverse geocoding (coordinates → address)

## Where the key lives: the SERVER, never the APK

Reverse geocoding runs **server-side** in `server/server.py`. The Google key is read from the
environment and never reaches the Android app, so it cannot be extracted from the APK by
anyone who installs it.

```bash
cd server
cp .env.example .env
```

Then edit `server/.env`:

```
GOOGLE_MAPS_API_KEY=your_key_here
```

`server/.env` is already covered by the repository `.gitignore` (`.env` and `.env.*`).
**Never commit a real key.**

## Google Cloud setup

1. Create/select a project in the Google Cloud console.
2. Enable **Geocoding API** (not "Maps SDK for Android" — that is a different product).
3. Create an API key.
4. Restrict it: **API restrictions → Geocoding API only**, and add an IP restriction for the
   machine running the receiver. An unrestricted key is a billing incident waiting to happen.
5. Billing must be enabled; the Geocoding API has no free-forever tier beyond the monthly credit.

## Without a key

The system degrades rather than breaking. `resolve_address()` returns `None`, the event keeps
its coordinates, and the dashboard shows the event with lat/lon and a map link but no street
name. `/ws/dashboard`'s greeting includes `"geocoding": false` so the UI can tell the
difference between "not configured" and "no result".

## When it is called — and when it is NOT

Reverse geocoding runs **only when a road event is generated**. It is never called for a GPS
packet: at 20 Hz that would be thousands of billed requests per minute.

## Caching and deduplication

Results are cached in-process, keyed on coordinates **rounded to 4 decimal places (~11 m)**:

```python
GEOCODE_PRECISION = 4
_address_cache: dict[tuple[float, float], str] = {}
```

So the same pothole reported from slightly different GPS samples, or by two devices, resolves
to one billed call. Verified: 4 lookups spanning identical coordinates, sub-metre jitter, and
a genuinely different location produce exactly **2** Google calls.

Combined with the app-side `perTypeCooldownMs` (8 s), a single defect cannot generate a stream
of API requests.

## Failure handling

- Network/timeout failure → logged, returns `None`, event still stored and broadcast.
- `ZERO_RESULTS` → returns `None` silently (a legitimate answer in open country).
- Any other status (`REQUEST_DENIED`, `OVER_QUERY_LIMIT`, …) → logged with Google's own
  `error_message`, because these are setup problems that are otherwise invisible.

The call runs via `asyncio.to_thread`, so a slow Google response never blocks the WebSocket
event loop that is receiving sensor packets.

## Resulting event

```json
{
  "type": "road_event",
  "event_id": "…",
  "event_type": "POTHOLE",
  "detection_label": "Pothole",
  "visual_confidence": 0.94,
  "confirmation_status": "CONFIRMED",
  "sensor_support": "STRONG",
  "shock_score": 12.2,
  "correlation_delta_ms": 1200,
  "latitude": 30.763940,
  "longitude": 76.573267,
  "address": "Sector 34, Chandigarh, 160022, India"
}
```
