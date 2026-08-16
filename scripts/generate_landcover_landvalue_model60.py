#!/usr/bin/env python3
"""Update the DINOv2 × land-value split map with model-60 median estimates.

The existing DINOv2 layer and split-view presentation are retained verbatim.
Only the right-hand land-value GeoJSON and its legend values are replaced.
"""

from pathlib import Path
import json
import os
import re

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pyproj import Transformer


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_HTML = (
    REPOSITORY / "landcover_landvalue" / "LandCover_DINO_LandValue_TabPFN.html"
)
INPUT_CSV = Path(
    "/Volumes/ssd_externo/projects_full/Tabular/modelos_newTPS/etapa4/"
    "resultados/inferencia_consolidado_cfg60_450m2.csv"
)

VALUE_COLUMN = "unit_q50"
GRID_SPACING_M = 109.45
HALF_GRID_M = GRID_SPACING_M / 2
REFERENCE_AREA_M2 = 450
MODEL_NUMBER = 60
DATA_YEAR = 2024
EXPECTED_CITIES = {
    "Apucarana", "Arapongas", "Cambe", "Cambira", "Ibipora", "Jandaia",
    "Londrina", "Mandaguari", "Marialva", "Maringa", "Rolandia", "Sarandi",
}

TIM_COLORS = [
    "#000080", "#0000CC", "#0040FF", "#0080FF", "#00BFFF", "#00FFFF",
    "#00FFBF", "#00FF80", "#00FF40", "#00FF00", "#40FF00", "#80FF00",
    "#BFFF00", "#FFFF00", "#FFBF00", "#FF8000", "#FF4000", "#FF0000",
]
COLOR_MAP = mcolors.LinearSegmentedColormap.from_list("tim", TIM_COLORS, N=1000)
PALETTE = [mcolors.rgb2hex(COLOR_MAP(i)) for i in np.linspace(0, 1, 1000)]
TO_WGS84 = Transformer.from_crs("EPSG:29192", "EPSG:4326", always_xy=True)


def validate(data: pd.DataFrame) -> None:
    required = {
        "utm_x", "utm_y", "cidade", "area_m2", "configuracao", VALUE_COLUMN,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    cities = set(data["cidade"].dropna().unique())
    if cities != EXPECTED_CITIES:
        raise ValueError(
            f"Expected 12 cities; missing={sorted(EXPECTED_CITIES-cities)}, "
            f"unexpected={sorted(cities-EXPECTED_CITIES)}"
        )
    if not data["area_m2"].eq(REFERENCE_AREA_M2).all():
        raise ValueError("The inference file does not exclusively use 450 m2.")
    if not data["configuracao"].eq(MODEL_NUMBER).all():
        raise ValueError("The inference file does not exclusively use model 60.")
    values = data[VALUE_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{VALUE_COLUMN} must contain finite positive values.")


def square_coordinates(east: float, north: float) -> list[list[list[float]]]:
    corners = (
        (east - HALF_GRID_M, north - HALF_GRID_M),
        (east + HALF_GRID_M, north - HALF_GRID_M),
        (east + HALF_GRID_M, north + HALF_GRID_M),
        (east - HALF_GRID_M, north + HALF_GRID_M),
        (east - HALF_GRID_M, north - HALF_GRID_M),
    )
    coordinates = []
    for x, y in corners:
        longitude, latitude = TO_WGS84.transform(x, y)
        coordinates.append([longitude, latitude])
    return [coordinates]


def build_landvalue_geojson(data: pd.DataFrame) -> tuple[dict, np.ndarray]:
    values = data[VALUE_COLUMN].to_numpy(dtype=float)
    log_values = np.log(values)
    minimum, maximum = float(log_values.min()), float(log_values.max())
    value_range = maximum - minimum
    if value_range <= 0:
        raise ValueError(f"{VALUE_COLUMN} has no variation.")

    features = []
    for east, north, value in data[["utm_x", "utm_y", VALUE_COLUMN]].itertuples(
        index=False, name=None
    ):
        normalized = (np.log(value) - minimum) / value_range
        color_index = int(np.clip(normalized, 0, 1) * (len(PALETTE) - 1))
        color = PALETTE[color_index]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": square_coordinates(float(east), float(north)),
                },
                "properties": {
                    "fillColor": color,
                    "color": color,
                    "weight": 0,
                    "fillOpacity": 0.7,
                    "price": f"R$ {float(value):,.2f}/m²",
                    "popup": (
                        f"Predicted Median Unit Land Value ({DATA_YEAR})<br>"
                        f"R$ {float(value):,.2f}/m²"
                    ),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}, log_values


def json_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        character = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
            if depth == 0:
                return position + 1
    raise ValueError("Could not find the end of the embedded JSON object.")


def replace_json_assignment(html: str, variable: str, value: dict) -> str:
    marker = f"var {variable} = "
    marker_position = html.find(marker)
    if marker_position < 0:
        raise ValueError(f"JavaScript variable {variable} was not found.")
    start = marker_position + len(marker)
    while start < len(html) and html[start].isspace():
        start += 1
    if start >= len(html) or html[start] not in "[{":
        raise ValueError(f"JavaScript variable {variable} does not contain JSON.")
    end = json_end(html, start)
    replacement = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return html[:start] + replacement + html[end:]


def replace_legend_values(html: str, log_values: np.ndarray) -> str:
    heading = "Median Unit Urban Land Value (TabPFN v2)"
    start = html.find(heading)
    if start < 0:
        raise ValueError("Land-value legend heading was not found.")
    end = html.find("Northern Paraná, Brazil • 2024", start)
    if end < 0:
        raise ValueError("Land-value legend footer was not found.")

    minimum, maximum = float(log_values.min()), float(log_values.max())
    positions = np.linspace(maximum, minimum, 5)
    labels = iter(f"R$ {np.exp(position):,.0f}" for position in positions)
    section = html[start:end]
    pattern = re.compile(r"<span>R\$\s*[0-9,]+</span>")
    if len(pattern.findall(section)) != 5:
        raise ValueError("Expected five numeric labels in the land-value legend.")
    section = pattern.sub(lambda _: f"<span>{next(labels)}</span>", section)
    return html[:start] + section + html[end:]


def count_embedded_features(html: str, variable: str) -> int:
    marker = f"var {variable} = "
    start = html.find(marker)
    if start < 0:
        raise ValueError(f"JavaScript variable {variable} was not found.")
    start += len(marker)
    while html[start].isspace():
        start += 1
    end = json_end(html, start)
    return len(json.loads(html[start:end])["features"])


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    validate(data)
    landvalue_geojson, log_values = build_landvalue_geojson(data)

    html = OUTPUT_HTML.read_text(encoding="utf-8")
    dino_features = count_embedded_features(html, "dataUso")
    html = replace_json_assignment(html, "dataValor", landvalue_geojson)
    html = replace_legend_values(html, log_values)

    temporary = OUTPUT_HTML.with_suffix(".tmp.html")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, OUTPUT_HTML)

    print(f"Validated {len(data):,} model-60 cells across 12 cities.")
    print(f"Preserved {dino_features:,} DINOv2 land-cover patches.")
    print(
        f"Generated {OUTPUT_HTML.name}: {OUTPUT_HTML.stat().st_size / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    main()
