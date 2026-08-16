#!/usr/bin/env python3
"""Generate the three public land-value maps from TabPFN model 60 estimates.

The presentation deliberately follows OLD_LandValuesMapsTabPFN.ipynb while
using the corrected CRS and the current 450 m2 model-60 inference results.
"""

from pathlib import Path
import os

import folium
from folium.features import GeoJsonTooltip
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pyproj import Transformer


REPOSITORY = Path(__file__).resolve().parents[1]
INPUT_CSV = Path(
    "/Volumes/ssd_externo/projects_full/Tabular/modelos_newTPS/etapa4/"
    "resultados/inferencia_consolidado_cfg60_450m2.csv"
)
OUTPUT_DIRECTORY = REPOSITORY / "landvalue"

GRID_SPACING_M = 109.45
HALF_GRID_M = GRID_SPACING_M / 2
PARADIGM_AREA_M2 = 450
DATA_YEAR = 2024
REFERENCE_DATE = "2026-08-01"
EXPECTED_CONFIGURATION = 60
EXPECTED_CITIES = {
    "Apucarana", "Arapongas", "Cambe", "Cambira", "Ibipora", "Jandaia",
    "Londrina", "Mandaguari", "Marialva", "Maringa", "Rolandia", "Sarandi",
}

# The old notebook used this approximation of R's tim.colors palette.
TIM_COLORS = [
    "#000080", "#0000CC", "#0040FF", "#0080FF", "#00BFFF", "#00FFFF",
    "#00FFBF", "#00FF80", "#00FF40", "#00FF00", "#40FF00", "#80FF00",
    "#BFFF00", "#FFFF00", "#FFBF00", "#FF8000", "#FF4000", "#FF0000",
]
COLOR_MAP = mcolors.LinearSegmentedColormap.from_list("tim", TIM_COLORS, N=1000)
PALETTE = [mcolors.rgb2hex(COLOR_MAP(i)) for i in np.linspace(0, 1, 1000)]

# The grid coordinates were derived in SAD69 / UTM zone 22S. The former
# notebook used EPSG:32722, which displaced the cells by about 66.5 m.
TO_WGS84 = Transformer.from_crs("EPSG:29192", "EPSG:4326", always_xy=True)


MAPS = (
    {
        "column": "unit_q10",
        "filename": "lowerBound_LandValueMapTabularPFN_NPr.html",
        "title": "Predicted Lower Bound Unit Value for Urban Land",
        "interval": True,
    },
    {
        "column": "unit_q50",
        "filename": "median_LandValueMapTabularPFN_NPr.html",
        "title": "Predicted Median Unit Value for Urban Land",
        "interval": False,
    },
    {
        "column": "unit_q90",
        "filename": "upperBound_LandValueMapTabularPFN_NPr.html",
        "title": "Predicted Upper Bound Unit Value for Urban Land",
        "interval": True,
    },
)


def validate(data: pd.DataFrame) -> None:
    required = {
        "utm_x", "utm_y", "cidade", "area_m2", "data_referencia",
        "configuracao", *(item["column"] for item in MAPS),
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    cities = set(data["cidade"].dropna().unique())
    if cities != EXPECTED_CITIES:
        raise ValueError(
            f"Expected the 12 site cities; missing={sorted(EXPECTED_CITIES-cities)}, "
            f"unexpected={sorted(cities-EXPECTED_CITIES)}"
        )
    if not data["area_m2"].eq(PARADIGM_AREA_M2).all():
        raise ValueError("The inference file does not exclusively use the 450 m2 parcel.")
    if not data["configuracao"].eq(EXPECTED_CONFIGURATION).all():
        raise ValueError("The inference file does not exclusively contain model 60.")
    if not data["data_referencia"].eq(REFERENCE_DATE).all():
        raise ValueError("Unexpected reference date in the inference file.")


def square_coordinates(x: float, y: float) -> list[list[list[float]]]:
    corners = (
        (x - HALF_GRID_M, y - HALF_GRID_M),
        (x + HALF_GRID_M, y - HALF_GRID_M),
        (x + HALF_GRID_M, y + HALF_GRID_M),
        (x - HALF_GRID_M, y + HALF_GRID_M),
        (x - HALF_GRID_M, y - HALF_GRID_M),
    )
    coordinates = []
    for east, north in corners:
        longitude, latitude = TO_WGS84.transform(east, north)
        coordinates.append([longitude, latitude])
    return [coordinates]


def legend_html(title: str, interval: bool, breaks: np.ndarray) -> str:
    heading = f"{title} (R$/m²: {DATA_YEAR})"
    if interval:
        heading += "<br>80% Prediction Interval"

    rows = []
    for index in range(len(breaks) - 1):
        color_index = int((index / (len(breaks) - 2)) * (len(PALETTE) - 1))
        lower, upper = np.exp(breaks[index:index + 2])
        rows.append(
            '<tr>'
            f'<td style="background-color:{PALETTE[color_index]}; width:20px; '
            'height:10px; border:1px solid #ccc;"></td>'
            '<td style="padding-left:8px; vertical-align:middle;">'
            f'R$ {lower:,.0f}/m² - R$ {upper:,.0f}/m²</td>'
            '</tr>'
        )

    return f"""
<div style="position: fixed;
           top: 10px; right: 10px; width: 280px; height: 570px;
           background-color: white; border:2px solid grey; z-index:9999;
           font-size:12px; padding: 10px; overflow-y: scroll;">
<p><b>{heading}</b></p>
<table style="font-size:10px; border-collapse: collapse;">
{''.join(rows)}
</table>
<div style="font-size:10px; margin-top:15px; color:#666; line-height:1.4;">
<p style="margin:0; font-weight:bold;">Paradigm:</p>
<ul style="margin:5px 0; padding-left:15px;">
    <li>Spatial Resolution: {GRID_SPACING_M:.1f}m × {GRID_SPACING_M:.1f}m</li>
    <li>Urban vacant parcel with paradigm area of {PARADIGM_AREA_M2}m²</li>
</ul>
</div>
</div>
"""


def generate_map(data: pd.DataFrame, specification: dict[str, object]) -> Path:
    column = str(specification["column"])
    title = str(specification["title"])
    values = data[column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{column} must contain only finite positive values.")

    log_values = np.log(values)
    minimum, maximum = float(log_values.min()), float(log_values.max())
    value_range = maximum - minimum
    if value_range <= 0:
        raise ValueError(f"{column} has no variation.")
    breaks = np.linspace(minimum, maximum, 18)

    center_x = float(data["utm_x"].mean())
    center_y = float(data["utm_y"].mean())
    center_lon, center_lat = TO_WGS84.transform(center_x, center_y)
    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
    )

    features = []
    interval_line = "<br>80% Prediction Interval" if specification["interval"] else ""
    for row in data[["utm_x", "utm_y", column]].itertuples(index=False, name=None):
        east, north, value = row
        normalized = (np.log(value) - minimum) / value_range
        color_index = int(np.clip(normalized, 0, 1) * (len(PALETTE) - 1))
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": square_coordinates(float(east), float(north)),
                },
                "properties": {
                    "fillColor": PALETTE[color_index],
                    "color": PALETTE[color_index],
                    "weight": 0,
                    "fillOpacity": 0.7,
                    "tooltip": (
                        f"{title}{interval_line}: "
                        f"R$ {float(value):,.0f}/m²"
                    ),
                },
            }
        )

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fillColor"],
            "color": feature["properties"]["color"],
            "weight": feature["properties"]["weight"],
            "fillOpacity": feature["properties"]["fillOpacity"],
            "opacity": 0,
        },
        tooltip=GeoJsonTooltip(fields=["tooltip"], aliases=[""], labels=False),
    ).add_to(map_object)

    map_object.get_root().html.add_child(
        folium.Element(legend_html(title, bool(specification["interval"]), breaks))
    )

    output = OUTPUT_DIRECTORY / str(specification["filename"])
    temporary = output.with_suffix(".tmp.html")
    map_object.save(str(temporary))
    os.replace(temporary, output)
    return output


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    validate(data)
    print(f"Validated {len(data):,} cells across {data['cidade'].nunique()} cities.")
    for specification in MAPS:
        output = generate_map(data, specification)
        print(f"Generated {output.name}: {output.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
