# IICB IPC Ring Map

A browser-based tool for mapping healthcare facilities around a case location,
with proximity filtering, data-quality deduplication, and optional
OpenStreetMap (OSM) facility overlay. Runs entirely in the browser via
[Shinylive](https://shiny.posit.co/py/docs/shinylive.html) — **no server, and
no facility data ever leaves the user's machine.**

## Live app

👉 **https://<user>.github.io/<repo>/**

## Features

- **Upload a facility CSV** and map your latitude, longitude, ID, and (optional)
  score and assessment-date columns.
- **Set a case location** (lat/lon) and a search **radius**; facilities within
  the radius are plotted and tabulated with computed distances.
- **Data-quality handling:**
  - Rows with invalid/blank coordinates are dropped (and reported).
  - Multiple rows for the same facility are collapsed to one — the **most recent
    assessment date** wins when a date column is mapped, otherwise the first
    occurrence. Facility counts reflect unique facilities, not raw rows.
- **Score coloring:** green ≥ 80, orange 50–79, red < 50, gray = unscored.
- **Column filters:** pick up to three numeric (range sliders) and three
  categorical (multi-select) columns to filter the map and table.
- **OSM overlay:** optionally fetch nearby hospitals/clinics/pharmacies/doctors
  from OpenStreetMap; markers co-located with an uploaded facility are removed.
- **Download** the currently filtered table as CSV.

## Using the app

1. **Facility Data** (sidebar): upload a CSV, click **Process**, and confirm the
   auto-detected ID / Latitude / Longitude / Score / Assessment date columns.
2. **Case Location** (above the map): enter Case ID, Latitude, Longitude, then
   **Set Location**.
3. **Map Options:** adjust radius, max score, and the OSM toggle.
4. **Filters:** choose columns to filter; use **Clear all filters** to reset.
5. Select table rows to highlight them on the map; use **Download filtered CSV**
   to export what's shown.

### Expected CSV format

Any CSV with at least numeric latitude and longitude columns. Optional columns
for a facility identifier, a numeric score (0–100), and an assessment date
(any format pandas can parse). Example:

| facility_id | name         | lat     | lon       | score | assessment_date |
|-------------|--------------|---------|-----------|-------|-----------------|
| F-001       | North Clinic | 39.7621 | -104.9812 | 72    | 2025-03-14      |

## Notes & limitations

- **First load is slow.** The browser downloads the Python runtime plus `osmnx`,
  `folium`, and their dependencies (tens of MB). Subsequent loads are cached.
- **OSM fetch briefly freezes the UI** in the browser build (Pyodide is
  single-threaded); a "loading…" indicator appears first. OSM data comes from
  the public Overpass API and is subject to its rate limits and availability.
- Scores are for internal use; the map is a decision aid, not an authoritative
  facility registry.

## Local development

```bash
pip install shiny osmnx folium
shiny run app/app.py --reload
```

The app runs identically locally (threaded OSM fetch) and in the browser
(synchronous fetch) from the same source file.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which runs
`shinylive export app _site` and publishes to GitHub Pages. Enable
**Settings → Pages → Source: GitHub Actions** once.