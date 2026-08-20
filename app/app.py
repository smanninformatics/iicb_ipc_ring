# app.py — Facility Proximity Mapper
# Runs on shinylive.io (Pyodide) AND server-hosted Shiny from the same file.
#
# Requires: shiny >= 1.0 (DataGrid selection + data_view), folium, osmnx
#   requirements.txt (second file on shinylive.io):
#       osmnx
#       folium
#
# NOTE (Shiny Express): any bare module-level expression that returns a
# non-None, non-tag value gets collected into the page UI and crashes
# rendering. Assign throwaway returns to `_`.
#
# UI convention: `ui.` (Express) for static top-level page structure;
# `core_ui.` for anything built and RETURNED inside a @render.* function.

import re
import sys
import math
import asyncio
from functools import lru_cache

import folium
import numpy as np
import osmnx as ox
import pandas as pd
from folium.plugins import MarkerCluster
from shiny import reactive
from shiny import ui as core_ui                       # returns HTML (for renders)
from shiny.express import input, render, ui           # context managers (page)

# ---------------------------------------------------------------- environment
IS_PYODIDE = sys.platform == "emscripten"             # True inside shinylive

if not IS_PYODIDE:
    import atexit
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)
    _ = atexit.register(executor.shutdown, wait=False)  # assign — Express collects
else:
    executor = None   # no OS threads in WASM; fetch runs synchronously instead

# ---------------------------------------------------------------- config
ox.settings.use_cache = True
ox.settings.cache_folder = "./cache"   # MEMFS in Pyodide: ephemeral per session
ox.settings.timeout = 30

CLUSTER_THRESHOLD = 400       # above this many markers, wrap in MarkerCluster
OSM_FETCH_RADIUS_KM = 20      # fetch once at max radius; filter reactively
PREVIEW_ROWS = 200
MAX_CATEGORY_LEVELS = 30      # text col with more uniques than this isn't "categorical"

# ---------------------------------------------------------------- helpers
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine (km). Supports numpy broadcasting."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def get_geometry_coords(geom):
    if geom.geom_type == "Point":
        return geom.y, geom.x
    pt = geom.representative_point()
    return pt.y, pt.x


def _guess(cols, *keywords):
    """Best-guess default for a column select by name substring."""
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return cols[0] if cols else None


def _slug(name: str) -> str:
    """Stable input-id fragment for an arbitrary column name."""
    return re.sub(r"\W+", "_", str(name)).strip("_").lower()


@lru_cache(maxsize=50)
def fetch_osm_cached(lat_rounded, lon_rounded, radius_m):
    """Fetch OSM healthcare POIs, cached by ~1km-rounded coordinates."""
    try:
        tags = {"amenity": ["hospital", "clinic", "doctors", "pharmacy"]}
        pois = ox.features_from_point((lat_rounded, lon_rounded),
                                      tags=tags, dist=radius_m)
        if pois.empty:
            return None
        coords = [get_geometry_coords(g) for g in pois.geometry]
        n = len(pois)
        return {
            "lats": np.array([c[0] for c in coords]),
            "lons": np.array([c[1] for c in coords]),
            "names": (pois["name"].fillna("Facility").values
                      if "name" in pois.columns else np.full(n, "Facility")),
            "amenities": (pois["amenity"].fillna("healthcare").values
                          if "amenity" in pois.columns
                          else np.full(n, "healthcare")),
        }
    except Exception:
        return None


def process_osm_result(osm_result, lat, lon, max_radius):
    if osm_result is None:
        return pd.DataFrame()
    distances = haversine(lat, lon, osm_result["lats"], osm_result["lons"])
    mask = distances <= max_radius
    return pd.DataFrame({
        "name": osm_result["names"][mask],
        "lat": osm_result["lats"][mask],
        "lon": osm_result["lons"][mask],
        "distance": distances[mask],
        "amenity": osm_result["amenities"][mask],
    })

def _swatch(color, label):
    return (f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
            f'white-space:nowrap;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
            f'display:inline-block;margin-right:5px;border:1px solid #555;"></span>'
            f'{label}</span>')

LEGEND_HTML = (
    '<div style="display:flex;flex-wrap:wrap;row-gap:4px;font-size:0.85em;'
    'margin-top:8px;border-top:1px solid #eee;padding-top:6px;">'
    + _swatch("#2e7d32", "Score ≥ 80")
    + _swatch("orange", "Score 50–79")
    + _swatch("red", "Score < 50")
    + _swatch("gray", "Unscored")
    + _swatch("cadetblue", "No score column")
    + _swatch("blue", "Case location")
    + _swatch("purple", "OSM facility")
    + "</div>"
)

# ---------------------------------------------------------------- state
facdf = reactive.value(pd.DataFrame())
case_location = reactive.value(None)   # {"lat","lon","id"} — set by button only
osm_cache = reactive.value({"lat": None, "lon": None,
                            "data": pd.DataFrame(), "loading": False})

# ---------------------------------------------------------------- calcs
@reactive.calc
def column_map():
    try:
        lat_col, lon_col = input.lat_col(), input.long_col()
    except Exception:                       # dynamic inputs not rendered yet
        return None
    if not lat_col or not lon_col:
        return None
    try:
        id_col = input.id_col()
    except Exception:
        id_col = None
    try:
        score_col = input.score_col()
    except Exception:
        score_col = None
    if score_col == "(none)":
        score_col = None
    try:
        date_col = input.date_col()
    except Exception:
        date_col = None
    if date_col == "(none)":
        date_col = None
    return {"id": id_col, "lat": lat_col, "lon": lon_col,
            "score": score_col, "date": date_col}


@reactive.calc
def cleaned_facility_data():
    """Valid-coordinate rows with distance. Returns (df, n_invalid_rows)."""
    df, loc, cols = facdf.get(), case_location.get(), column_map()
    if df.empty or loc is None or cols is None:
        return pd.DataFrame(), 0
    out = df.copy()
    out[cols["lat"]] = pd.to_numeric(out[cols["lat"]], errors="coerce")
    out[cols["lon"]] = pd.to_numeric(out[cols["lon"]], errors="coerce")
    valid = out[cols["lat"]].notna() & out[cols["lon"]].notna()
    n_invalid = int((~valid).sum())
    out = out[valid]
    if out.empty:
        return out, n_invalid
    out["distance"] = haversine(loc["lat"], loc["lon"],
                                out[cols["lat"]].values,
                                out[cols["lon"]].values)
    return out, n_invalid


@reactive.calc
def deduplicated_data():
    """One row per facility (by ID). When a date column is mapped, the most
    recent assessment wins; otherwise the first occurrence is kept.
    Returns (df, n_duplicate_rows_collapsed)."""
    df, _ = cleaned_facility_data()
    if df.empty:
        return df, 0
    cols = column_map() or {}
    id_col = cols.get("id")
    if not id_col or id_col not in df.columns:
        return df, 0                       # no ID → nothing to dedup on
    work = df.copy()
    date_col = cols.get("date")
    if date_col and date_col in work.columns:
        work["_asmt_date"] = pd.to_datetime(work[date_col], errors="coerce")
        # most-recent first; NaT sorts last so a dated row beats an undated one
        work = work.sort_values("_asmt_date", ascending=False, na_position="last")
    n_before = len(work)
    work = work.drop_duplicates(subset=[id_col], keep="first")
    n_dupes = n_before - len(work)
    return work.drop(columns=["_asmt_date"], errors="ignore"), n_dupes


@reactive.calc
def get_ring_data():
    """Deduped facilities within radius, under max score (parent constraints)."""
    df, _ = deduplicated_data()
    if df.empty:
        return df
    cols = column_map()
    mask = df["distance"] <= input.radius()
    if cols and cols["score"] and cols["score"] in df.columns:
        mask &= (pd.to_numeric(df[cols["score"]], errors="coerce")
                 <= float(input.score_max()))
    out = df[mask]
    return out.assign(distance=out["distance"].round(2))


@reactive.calc
def available_numeric_cols():
    """Numeric columns offered in the numeric-filter picker (deduped bounds)."""
    df, _ = deduplicated_data()
    if df.empty:
        return []
    cols = column_map() or {}
    skip = {cols.get("lat"), cols.get("lon"), cols.get("id"),
            cols.get("date"), "distance"}
    out = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in skip:
            continue
        if pd.to_numeric(df[c], errors="coerce").dropna().nunique() > 1:
            out.append(c)
    return out


@reactive.calc
def available_categorical_cols():
    """Low-cardinality text columns offered in the category-filter picker."""
    df, _ = deduplicated_data()
    if df.empty:
        return []
    cols = column_map() or {}
    skip = {cols.get("lat"), cols.get("lon"), cols.get("id"),
            cols.get("date"), "distance"}
    out = []
    for c in df.select_dtypes(include=["object", "string", "category"]).columns:
        if c in skip:
            continue
        if 1 < df[c].nunique(dropna=True) <= MAX_CATEGORY_LEVELS:
            out.append(c)
    return out


@reactive.calc
def display_data():
    """Ring data after the picked column filters. Drives map/stats/download."""
    df = get_ring_data()
    if df.empty:
        return df

    # categorical multi-selects
    for c in (input.cat_picker() or ()):
        try:
            sel = input[f"cat_{_slug(c)}"]()
        except Exception:
            continue
        if sel:
            df = df[df[c].astype(str).isin(list(sel))]

    # numeric range sliders (keep NaN so an untouched filter never drops rows
    # for missing values in an unrelated column)
    for c in (input.num_picker() or ()):
        try:
            lo, hi = input[f"num_{_slug(c)}"]()
        except Exception:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        df = df[v.between(lo, hi) | v.isna()]

    return df


@reactive.calc
def plotted_osm_data():
    """OSM rows actually drawn: within radius AND not co-located (<100m) with a
    displayed facility. Both the map and the stats read this, so counts match."""
    osm = get_osm_data()
    if osm.empty:
        return osm
    ring = display_data()
    cols = column_map() or {}
    if ring.empty or not cols.get("lat") or cols["lat"] not in ring.columns:
        return osm
    lats = ring[cols["lat"]].values
    lons = ring[cols["lon"]].values
    if len(lats) == 0:
        return osm
    olats, olons = osm["lat"].values, osm["lon"].values
    d = haversine(olats[:, None], olons[:, None], lats[None, :], lons[None, :])
    return osm[d.min(axis=1) >= 0.1]


@reactive.calc
def selected_rows():
    try:
        return results.data_view(selected=True)
    except Exception:
        return pd.DataFrame()


@reactive.calc
def get_osm_data():
    if not input.osm():
        return pd.DataFrame()
    df = osm_cache.get()["data"]
    if df.empty:
        return df
    return df[df["distance"] <= input.radius()]


@reactive.calc
def osm_loading():
    return osm_cache.get()["loading"]

# ---------------------------------------------------------------- OSM async
@reactive.extended_task
async def fetch_osm_async(lat, lon, max_radius):
    lat_r, lon_r = round(lat, 2), round(lon, 2)
    radius_m = int(max_radius * 1100)
    if IS_PYODIDE:
        # Single-threaded WASM: yield so 'loading…' paints before we block.
        await asyncio.sleep(0.05)
        result = fetch_osm_cached(lat_r, lon_r, radius_m)
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor, fetch_osm_cached, lat_r, lon_r, radius_m)
    # Coordinates travel WITH the result — fixes stale-cache race on location change.
    return lat, lon, process_osm_result(result, lat, lon, max_radius)


@reactive.effect
def update_osm_cache():
    res = fetch_osm_async.result()
    if res is not None:
        lat, lon, df = res
        osm_cache.set({"lat": lat, "lon": lon, "data": df, "loading": False})


@reactive.effect
@reactive.event(case_location, input.osm)
def trigger_osm_fetch():
    if not input.osm():
        return
    loc = case_location.get()
    if loc is None:
        return
    cache = osm_cache.get()
    if (cache["lat"] == loc["lat"] and cache["lon"] == loc["lon"]
            and not cache["data"].empty):
        return
    osm_cache.set({"lat": loc["lat"], "lon": loc["lon"],
                   "data": pd.DataFrame(), "loading": True})
    fetch_osm_async(loc["lat"], loc["lon"], OSM_FETCH_RADIUS_KM)


@reactive.effect
@reactive.event(input.refresh_osm)
def manual_osm_refresh():
    loc = case_location.get()
    if loc is None:
        return
    fetch_osm_cached.cache_clear()
    osm_cache.set({"lat": None, "lon": None,
                   "data": pd.DataFrame(), "loading": True})
    fetch_osm_async(loc["lat"], loc["lon"], OSM_FETCH_RADIUS_KM)

# ---------------------------------------------------------------- workflow
@reactive.effect
@reactive.event(input.process)
def load_data():
    fi = input.csv()
    if not fi:
        ui.notification_show("Choose a CSV file first.", type="warning")
        return
    try:
        df = pd.read_csv(fi[0]["datapath"])
    except Exception as e:
        ui.notification_show(f"Could not read CSV: {e}", type="error")
        return
    facdf.set(df)
    ui.notification_show(
        f"Loaded {len(df):,} rows × {df.shape[1]} columns. "
        "Now set the case location above the map.", duration=6)


@reactive.effect
@reactive.event(input.save_case_headers)
def save_location():
    try:
        lat, lon = float(input.lat()), float(input.long())
    except (ValueError, TypeError):
        ui.notification_show("Enter valid numeric coordinates.", type="error")
        return
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        ui.notification_show("Coordinates out of range.", type="error")
        return
    case_location.set({"lat": lat, "lon": lon, "id": (input.case_id() or "Case")})


@reactive.effect
def _populate_pickers():
    """Refill filter-column pickers whenever the underlying data changes."""
    ui.update_selectize("num_picker", choices=available_numeric_cols())
    ui.update_selectize("cat_picker", choices=available_categorical_cols())


@reactive.effect
@reactive.event(input.clear_filters)
def clear_all_filters():
    ui.update_selectize("num_picker", selected=[])
    ui.update_selectize("cat_picker", selected=[])

# ---------------------------------------------------------------- UI
ui.page_opts(title="IPC Ring Mapper", fillable=False)

# ---- sidebar: facility data only ----
with ui.sidebar(width=340):
    ui.h5("Facility Data")
    ui.input_file("csv", None, accept=".csv")
    ui.input_action_button("process", "Process", class_="btn-primary w-100")

    @render.ui
    @reactive.event(input.process)
    def controls_mapping():
        df = facdf.get()
        if df.empty:
            return None
        cols = df.columns.tolist()
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        return core_ui.TagList(
            core_ui.hr(),
            core_ui.input_select("id_col", "ID column", cols,
                                 selected=_guess(cols, "id", "name")),
            core_ui.input_select("lat_col", "Latitude", nums,
                                 selected=_guess(nums, "lat")),
            core_ui.input_select("long_col", "Longitude", nums,
                                 selected=_guess(nums, "lon", "lng")),
            core_ui.input_select("score_col", "Score (optional)",
                                 ["(none)"] + nums),
            core_ui.input_select("date_col", "Assessment date (optional)",
                                 ["(none)"] + cols,
                                 selected=_guess(["(none)"] + cols,
                                                 "date", "assess")),
        )

    @render.text
    def file_summary():
        df = facdf.get()
        if df.empty:
            return "No file loaded."
        return f"{len(df):,} rows × {df.shape[1]} columns loaded."

# ---- case location + map options, visible above the map ----
with ui.card(fill=False, class_="mb-3"):
    ui.card_header("Case Location")
    with ui.layout_column_wrap(width="240px", fill=False):
        ui.input_text("case_id", "Case ID", placeholder="e.g. C-1042")
        ui.input_text("lat", "Latitude", placeholder="39.7392")
        ui.input_text("long", "Longitude", placeholder="-104.9903")
    ui.input_action_button("save_case_headers", "Set Location",
                           class_="btn-primary", width="220px")

with ui.card(fill=False, class_="mb-3"):
    ui.card_header("Map Options")
    with ui.layout_column_wrap(width="280px", fill=False):
        ui.input_slider("radius", "Radius (km)", 1, OSM_FETCH_RADIUS_KM, 5)
        ui.input_slider("score_max", "Max facility score", 0, 100, 100)
        with ui.div(class_="d-flex flex-column justify-content-center h-100"):
            ui.input_checkbox("osm", "Include OSM facilities", True)
            ui.input_action_button("refresh_osm", "Refresh OSM",
                                   class_="btn-sm btn-secondary mt-2")

# ---- summary + legend (own card so map re-renders don't flash it) ----
with ui.card(fill=False, class_="mb-3"):
    ui.card_header("Summary")

    @render.text
    def mapping_stats():
        loc = case_location.get()
        if loc is None:
            return ("Set a case location above to draw the ring and show nearby "
                    "OSM facilities. Uploading facility data is optional.")
        osm_status = ("loading…" if osm_loading()
                      else str(len(plotted_osm_data())))
        cols = column_map()
        if cols is None or facdf.get().empty:      # OSM-only mode
            return (f"No facility file loaded — showing OSM facilities only. | "
                    f"OSM on map: {osm_status}")
        try:
            _, n_invalid = cleaned_facility_data()
            deduped, n_dupes = deduplicated_data()
            notes = []
            if n_invalid:
                notes.append(f"{n_invalid} invalid-coordinate rows dropped")
            if n_dupes:
                notes.append(f"{n_dupes} duplicate facility rows collapsed")
            note = f" ({'; '.join(notes)})" if notes else ""
            return (f"Facilities: {len(deduped)}{note} | "
                    f"In radius: {len(get_ring_data())} | "
                    f"On map (after filters): {len(display_data())} | "
                    f"OSM on map: {osm_status}")
        except Exception:
            return "Updating…"
            
    ui.HTML(LEGEND_HTML)

# ---- map ----
with ui.card(full_screen=True, height="560px"):
    ui.card_header("Map")

    @render.ui
    def map_display():
        loc = case_location.get()
        if loc is None:
            return core_ui.div(
                "The map will appear once a case location is set.",
                class_="text-muted p-3")
        cols = column_map()   # may be None when no facility file is loaded — that's fine
        try:
            m = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=12)
            folium.Marker([loc["lat"], loc["lon"]], tooltip=loc["id"],
                          icon=folium.Icon(color="blue", icon="user")).add_to(m)
            folium.Circle([loc["lat"], loc["lon"]],
                          radius=input.radius() * 1000,
                          color="red", fill=False).add_to(m)

            # ---- uploaded facilities (only when a file is mapped) ----
            ring = display_data()
            if (cols is not None and not ring.empty
                    and cols["lat"] in ring.columns and "distance" in ring.columns):
                lats = ring[cols["lat"]].values
                lons_arr = ring[cols["lon"]].values
                dists = ring["distance"].values
                ids = (ring[cols["id"]].values
                       if cols["id"] and cols["id"] in ring.columns
                       else np.arange(len(ring)))

                if cols["score"] and cols["score"] in ring.columns:
                    scores = pd.to_numeric(ring[cols["score"]],
                                           errors="coerce").values
                    colors = np.select(
                        [pd.isna(scores), scores < 50, scores < 80],
                        ["gray", "red", "orange"], default="green")
                    labels = [
                        (f"{ids[i]}: {scores[i]:.1f} ({dists[i]:.2f} km)"
                         if not pd.isna(scores[i])
                         else f"{ids[i]} — unscored ({dists[i]:.2f} km)")
                        for i in range(len(ring))]
                else:
                    colors = np.full(len(ring), "cadetblue")
                    labels = [f"{ids[i]} ({dists[i]:.2f} km)"
                              for i in range(len(ring))]

                container = (MarkerCluster().add_to(m)
                             if len(ring) > CLUSTER_THRESHOLD else m)
                for i in range(len(ring)):
                    folium.CircleMarker(
                        [lats[i], lons_arr[i]], radius=6, color=colors[i],
                        fill=True, fill_opacity=0.85, tooltip=labels[i],
                    ).add_to(container)

                # table row selection -> bold highlight ring
                sel = selected_rows()
                if not sel.empty and cols["lat"] in sel.columns:
                    for _, row in sel.iterrows():
                        folium.CircleMarker(
                            [row[cols["lat"]], row[cols["lon"]]],
                            radius=11, color="#000", weight=3, fill=False,
                            tooltip="Selected in table",
                        ).add_to(m)

            # ---- OSM facilities (always drawn when a location is set) ----
            for _, row in plotted_osm_data().iterrows():
                folium.CircleMarker(
                    [row["lat"], row["lon"]], radius=5, color="purple",
                    fill=True, fill_opacity=0.7,
                    tooltip=f"OSM: {row['name']} ({row['amenity']})",
                ).add_to(m)

            return core_ui.HTML(m._repr_html_())
        except Exception as e:
            import traceback
            return core_ui.div(f"Error creating map: {e}",
                               core_ui.tags.pre(traceback.format_exc()))
            
# ---- filters + table ----
with ui.card(full_screen=True):                 # no fixed height → no overlap
    ui.card_header("Facilities")

    with ui.accordion(id="filter_panel", open="filters"):
        with ui.accordion_panel("🔍 Filters", value="filters"):
            with ui.layout_columns(col_widths=[6, 6]):
                ui.input_selectize("num_picker", "Numeric filters (max 3)",
                                   choices=[], multiple=True,
                                   options={"maxItems": 3,
                                            "placeholder": "select columns…"})

                ui.input_selectize("cat_picker", "Category filters (max 3)",
                                   choices=[], multiple=True,
                                   options={"maxItems": 3,
                                            "placeholder": "select columns…"})
    
                @render.ui
                def numeric_filters():
                    picks = input.num_picker() or ()
                    if not picks:
                        return None
                    df, _ = deduplicated_data()
                    items = []
                    with reactive.isolate():
                        for c in picks:
                            s = pd.to_numeric(df[c], errors="coerce").dropna()
                            if s.empty:
                                continue
                            lo = math.floor(float(s.min()))
                            hi = math.ceil(float(s.max()))
                            if hi <= lo:
                                hi = lo + 1
                            try:
                                cur = input[f"num_{_slug(c)}"]()
                                val = (max(lo, min(cur[0], hi)),
                                       min(hi, max(cur[1], lo)))
                            except Exception:
                                val = (lo, hi)
                            items.append(core_ui.div(
                                core_ui.input_slider(f"num_{_slug(c)}", c,
                                                     min=lo, max=hi, value=val),
                                style="flex:1 1 260px; min-width:260px;"))
                    return core_ui.div(*items, class_="d-flex flex-wrap gap-3")
            
                @render.ui
                def category_filters():
                    picks = input.cat_picker() or ()
                    if not picks:
                        return None
                    df, _ = deduplicated_data()
                    items = []
                    with reactive.isolate():
                        for c in picks:
                            choices = sorted(df[c].dropna().astype(str)
                                             .unique().tolist())
                            try:
                                prior = list(input[f"cat_{_slug(c)}"]())
                            except Exception:
                                prior = []
                            items.append(core_ui.div(
                                core_ui.input_selectize(
                                    f"cat_{_slug(c)}", c, choices=choices,
                                    multiple=True, selected=prior,
                                    options={"placeholder": "All values"}),
                                style="flex:1 1 240px; min-width:240px;"))
                    return core_ui.div(*items, class_="d-flex flex-wrap gap-3")
    
                ui.input_action_button("clear_filters", "Clear all filters",
                                       class_="btn-warning btn-sm mt-2")

            @render.text
            def filter_summary():
                n_ring, n_show = len(get_ring_data()), len(display_data())
                bits = []
                for c in (input.cat_picker() or ()):
                    try:
                        sel = input[f"cat_{_slug(c)}"]()
                    except Exception:
                        continue
                    if sel:
                        bits.append(f"{c}: {len(sel)} selected")
                for c in (input.num_picker() or ()):
                    try:
                        lo, hi = input[f"num_{_slug(c)}"]()
                    except Exception:
                        continue
                    bits.append(f"{c}: {lo:g}–{hi:g}")
                active = " · ".join(bits) if bits else "no column filters active"
                return (f"Showing {n_show} of {n_ring} in-radius facilities "
                        f"— {active}")

    # Download button in its own right-aligned row ABOVE the table (normal
    # document flow, so it can't overlap the grid rows).
    with ui.div(class_="d-flex justify-content-end mb-2"):
        @render.download(label="⬇ Download filtered CSV",
                         filename="facilities_filtered.csv")
        def dl():
            yield display_data().to_csv(index=False)

    @render.data_frame
    def results():
        if case_location.get() is None:
            df = facdf.get()
            return render.DataGrid(df.head(PREVIEW_ROWS), height="360px",
                                   width="100%")
        df = display_data()
        if df.empty:
            return render.DataGrid(pd.DataFrame())
        cols = column_map()
        front = [c for c in [cols["id"], cols["score"], cols["date"], "distance"]
                 if c and c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]
        return render.DataGrid(df, filters=False, height="360px",
                               width="100%", selection_mode="rows")