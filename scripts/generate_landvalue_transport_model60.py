#!/usr/bin/env python3
"""Update the land-value × transport HTML with TabPFN model-60 estimates.

The embedded road hierarchy and urban perimeters are retained. Only the median
land-value GeoJSON, its style function, and its legend are replaced.
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
    REPOSITORY / "landvalue_transport" / "LandValue_TabPFN_TransportStructure.html"
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
CITY_LABELS = {
    "Apucarana": "Apucarana", "Arapongas": "Arapongas", "Cambe": "Cambé",
    "Cambira": "Cambira", "Ibipora": "Ibiporã", "Jandaia": "Jandaia do Sul",
    "Londrina": "Londrina", "Mandaguari": "Mandaguari", "Marialva": "Marialva",
    "Maringa": "Maringá", "Rolandia": "Rolândia", "Sarandi": "Sarandi",
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
    columns = ["utm_x", "utm_y", "cidade", VALUE_COLUMN]
    for east, north, city, value in data[columns].itertuples(index=False, name=None):
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
                    "popup": (
                        "Predicted Median Unit Value for Urban Land at "
                        f"{CITY_LABELS[city]} ({DATA_YEAR}): R$ {float(value):,.0f}/m²"
                    ),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}, log_values


def structured_end(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for position in range(start, len(text)):
        character = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                in_string = False
            continue
        if character in {'"', "'", "`"}:
            in_string = True
            quote = character
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return position + 1
    raise ValueError("Could not find the end of the structured block.")


def embedded_geojson_layers(html: str) -> list[dict]:
    layers = []
    pattern = re.compile(r"(geo_json_[A-Za-z0-9]+)_add\s*\(")
    for match in pattern.finditer(html):
        start = match.end()
        while start < len(html) and html[start].isspace():
            start += 1
        if start >= len(html) or html[start] != "{":
            continue
        end = structured_end(html, start, "{", "}")
        value = json.loads(html[start:end])
        layers.append(
            {"name": match.group(1), "start": start, "end": end, "value": value}
        )
    return layers


def find_landvalue_layer(html: str) -> dict:
    candidates = []
    for layer in embedded_geojson_layers(html):
        features = layer["value"].get("features", [])
        if not features:
            continue
        popup = features[0].get("properties", {}).get("popup", "")
        if str(popup).startswith("Predicted Median Unit Value for Urban Land"):
            candidates.append(layer)
    if len(candidates) != 1:
        raise ValueError(f"Expected one land-value layer, found {len(candidates)}.")
    return candidates[0]


def replace_style_function(html: str, layer_name: str) -> str:
    function_name = f"function {layer_name}_styler(feature)"
    start = html.find(function_name)
    if start < 0:
        raise ValueError("The land-value style function was not found.")
    brace = html.find("{", start + len(function_name))
    end = structured_end(html, brace, "{", "}")
    replacement = f"""function {layer_name}_styler(feature) {{
            return {{
                "fillColor": feature.properties.fillColor,
                "color": feature.properties.color,
                "weight": feature.properties.weight,
                "fillOpacity": feature.properties.fillOpacity,
                "opacity": 0
            }};
        }}"""
    return html[:start] + replacement + html[end:]


def replace_landvalue_layer(html: str, geojson: dict) -> str:
    layer = find_landvalue_layer(html)
    replacement = json.dumps(geojson, ensure_ascii=False, separators=(",", ":"))
    html = html[:layer["start"]] + replacement + html[layer["end"]:]
    return replace_style_function(html, layer["name"])


def replace_legend(html: str, log_values: np.ndarray) -> str:
    heading = "Predicted Median Unit Value for Urban Land (R$/m²: 2024)"
    start = html.find(heading)
    if start < 0:
        raise ValueError("The land-value legend was not found.")
    end = html.find("Urban Transportation Network Hierarchy", start)
    if end < 0:
        raise ValueError("The transport legend was not found.")
    section = html[start:end]

    breaks = np.linspace(float(log_values.min()), float(log_values.max()), 18)
    ranges = [
        f"R$ {np.exp(breaks[i]):,.0f}/m² - R$ {np.exp(breaks[i+1]):,.0f}/m²"
        for i in range(len(breaks) - 1)
    ]
    iterator = iter(ranges)
    pattern = re.compile(r"R\$ [0-9,]+/m² - R\$ [0-9,]+/m²")
    if len(pattern.findall(section)) != len(ranges):
        raise ValueError("Expected 17 value ranges in the land-value legend.")
    section = pattern.sub(lambda _: next(iterator), section)
    if section.count("363m²") != 1:
        raise ValueError("Expected one old paradigm-area label.")
    section = section.replace("363m²", f"{REFERENCE_AREA_M2}m²")
    return html[:start] + section + html[end:]


def non_value_layer_fingerprints(html: str) -> dict[str, str]:
    import hashlib

    value_layer = find_landvalue_layer(html)["name"]
    result = {}
    for layer in embedded_geojson_layers(html):
        if layer["name"] == value_layer:
            continue
        canonical = json.dumps(
            layer["value"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        result[layer["name"]] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    validate(data)
    geojson, log_values = build_landvalue_geojson(data)

    html = OUTPUT_HTML.read_text(encoding="utf-8")
    preserved_before = non_value_layer_fingerprints(html)
    html = replace_landvalue_layer(html, geojson)
    html = replace_legend(html, log_values)
    preserved_after = non_value_layer_fingerprints(html)
    if preserved_before != preserved_after:
        raise RuntimeError("A perimeter or transport layer changed unexpectedly.")

    temporary = OUTPUT_HTML.with_suffix(".tmp.html")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, OUTPUT_HTML)

    print(f"Validated {len(data):,} model-60 cells across 12 cities.")
    print(f"Preserved {len(preserved_after)} perimeter/transport GeoJSON layers.")
    print(
        f"Generated {OUTPUT_HTML.name}: {OUTPUT_HTML.stat().st_size / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    main()
