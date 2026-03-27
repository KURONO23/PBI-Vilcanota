# ============================================================
# TÍTULO: PISCO_HyD_ARNOVIC + PBI (STREAMLIT LISTO PARA DEPLOY)
# - Funciona local y en Streamlit Community Cloud
# - Usa rutas relativas al repositorio
# - Lee secretos desde st.secrets en la nube
# - Si existe google-service.json local, también puede usarlo
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional
import io

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st
import folium

from streamlit_folium import st_folium
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pyproj import CRS
from branca.element import Element


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="PISCO_HyD_ARNOVIC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPORTANTE:
# En GitHub / Streamlit Cloud, BASE_DIR será la carpeta del repo.
BASE_DIR = Path(__file__).resolve().parent

# Opción local (NO subir este archivo a GitHub)
JSON_GOOGLE = BASE_DIR / "google-service.json"

DRIVE_FOLDER_ID = "176HOIc10u-_d-sZUkF94zmjgQi3CDYXM"

DATA_DIR = BASE_DIR / "tmp_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HIST_PATH = DATA_DIR / "hist_filtrado.parquet"
FORE_PATH = DATA_DIR / "fore_filtrado.parquet"

SHAPE_BASE = BASE_DIR / "shape"
SHP_ESTACIONES = SHAPE_BASE / "Estacion.shp"

ZONE_DIRS = {
    "utm17s": SHAPE_BASE / "utm17s",
    "utm18s": SHAPE_BASE / "utm18s",
    "utm19s": SHAPE_BASE / "utm19s",
}

AOI_LON = -72.06
AOI_LAT = -13.3364
AOI_ZOOM = 13
CRS_WGS = 4326
CRS_METRICO = 32719

REF_YEAR = 1999  # año de referencia para superponer septiembre-agosto


# ============================================================
# ESTILO
# ============================================================
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .titulo-app {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: .15rem;
    }
    .subtitulo-app {
        color: #475569;
        margin-bottom: 1rem;
    }
    .fecha-box {
        background: #facc15;
        color: #111827;
        font-weight: 800;
        padding: 10px 14px;
        border-radius: 0px;
        display: inline-block;
        margin-bottom: .75rem;
    }
    .nivel-box {
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 1.8rem;
        color: white;
        margin-bottom: .5rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 900;
        letter-spacing: .3px;
        color: #1f2937;
        border-left: 6px solid #2582d8;
        padding-left: 10px;
        margin: 4px 0 10px 0;
    }
    .pbi-right-card {
        border: 1px solid #d9dee7;
        border-radius: 14px;
        background: #ffffff;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }
    .pbi-right-title {
        text-align: center;
        font-weight: 900;
        letter-spacing: 2px;
        color: #2c3e50;
        margin-bottom: 8px;
    }
    .exp-grid {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 8px;
    }
    .exp-row {
        display: grid;
        grid-template-columns: 1fr 110px;
        gap: 10px;
        align-items: stretch;
    }
    .exp-left, .exp-right {
        border: 2px solid #6b6f76;
        background: #fff;
        border-radius: 0px;
    }
    .exp-left {
        padding: 8px 10px;
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .exp-ico {
        width: 48px;
        min-width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        line-height: 1;
    }
    .exp-txt {
        line-height: 1.05;
    }
    .exp-name {
        font-weight: 900;
        font-size: 13px;
        color: #111;
    }
    .exp-unit {
        font-weight: 900;
        font-size: 12px;
        color: #111;
        opacity: .9;
    }
    .exp-right {
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 34px;
        color: #777;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS GENERALES
# ============================================================
def norm_name(x: str) -> str:
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def find_col(columns, targets) -> Optional[str]:
    cols_norm = [norm_name(c) for c in columns]
    tars_norm = [norm_name(t) for t in targets]

    for t in tars_norm:
        if t in cols_norm:
            return columns[cols_norm.index(t)]

    for t in tars_norm:
        for i, c in enumerate(cols_norm):
            if t in c:
                return columns[i]
    return None


def detect_longlat_bounds(bounds) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return (
        pd.notna([xmin, ymin, xmax, ymax]).all()
        and -180 <= xmin <= 180
        and -180 <= xmax <= 180
        and -90 <= ymin <= 90
        and -90 <= ymax <= 90
        and xmin < xmax
        and ymin < ymax
    )


def bounds_reasonable(bounds, max_span_deg=8) -> bool:
    xmin, ymin, xmax, ymax = bounds
    if not detect_longlat_bounds(bounds):
        return False
    return abs(xmax - xmin) < max_span_deg and abs(ymax - ymin) < max_span_deg


def epsg_from_zona_utm(zona_val) -> Optional[int]:
    if zona_val is None or pd.isna(zona_val):
        return None
    s = str(zona_val).strip().upper().replace(" ", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    z = int(digits)
    if 17 <= z <= 22:
        return 32700 + z
    return None


def epsg_from_lon(lon: float) -> Optional[int]:
    if lon is None or pd.isna(lon):
        return None
    zone = int((float(lon) + 180) // 6) + 1
    if 17 <= zone <= 22:
        return 32700 + zone
    return None


def metric_crs_for_gdf(gdf: Optional[gpd.GeoDataFrame], fallback_epsg: int = CRS_METRICO) -> int:
    if gdf is None or len(gdf) == 0:
        return fallback_epsg

    gdf = coerce_crs_safely(gdf)

    if "ZONA_UTM" in gdf.columns:
        vals = gdf["ZONA_UTM"].dropna().astype(str).tolist()
        if vals:
            epsg = epsg_from_zona_utm(vals[0])
            if epsg is not None:
                return epsg

    try:
        g_wgs = gdf.to_crs(4326) if (gdf.crs and gdf.crs.to_epsg() != 4326) else gdf
        c = g_wgs.unary_union.centroid
        epsg = epsg_from_lon(c.x)
        if epsg is not None:
            return epsg
    except Exception:
        pass

    return fallback_epsg



def normalize_zone_value(zona_val) -> Optional[str]:
    if zona_val is None or pd.isna(zona_val):
        return None
    s = str(zona_val).strip().lower().replace(" ", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits not in {"17", "18", "19"}:
        return None
    return f"utm{digits}s"


def zone_folder_from_lon(lon: float) -> Optional[str]:
    if lon is None or pd.isna(lon):
        return None
    zone = int((float(lon) + 180) // 6) + 1
    if zone in [17, 18, 19]:
        return f"utm{zone}s"
    return None


def zone_folder_from_gdf(gdf: Optional[gpd.GeoDataFrame]) -> Optional[str]:
    if gdf is None or len(gdf) == 0:
        return None

    gdf = coerce_crs_safely(gdf)

    if "ZONA_UTM" in gdf.columns:
        vals = gdf["ZONA_UTM"].dropna().astype(str).tolist()
        if vals:
            zone_folder = normalize_zone_value(vals[0])
            if zone_folder is not None:
                return zone_folder

    try:
        g_wgs = gdf.to_crs(4326) if (gdf.crs and gdf.crs.to_epsg() != 4326) else gdf
        c = g_wgs.unary_union.centroid
        return zone_folder_from_lon(c.x)
    except Exception:
        return None


def zone_dir_from_key(zone_key: str) -> Path:
    zone_key = normalize_zone_value(zone_key)
    if zone_key is None or zone_key not in ZONE_DIRS:
        raise ValueError(f"Zona UTM no válida: {zone_key}")
    return ZONE_DIRS[zone_key]


def inundacion_filename_for_zone(zone_key: str) -> str:
    zone_key = normalize_zone_value(zone_key)
    if zone_key is None:
        raise ValueError("No se pudo resolver la zona UTM para el shape de inundación.")
    suffix = zone_key.replace("utm", "UTM")
    return f"Inundacion_{suffix}.shp"


def zone_folder_for_comid(estaciones_gdf: gpd.GeoDataFrame, comid_sel: float) -> Optional[str]:
    if estaciones_gdf is None or len(estaciones_gdf) == 0 or pd.isna(comid_sel):
        return None

    sub = estaciones_gdf[pd.to_numeric(estaciones_gdf["COMID"], errors="coerce") == float(comid_sel)].copy()
    if len(sub) == 0:
        return None

    if "ZONA_FOLDER" in sub.columns:
        zf = sub["ZONA_FOLDER"].dropna().astype(str).tolist()
        if zf:
            return zf[0]

    return zone_folder_from_gdf(sub)


def guess_peru_utm(gdf: gpd.GeoDataFrame, zones=range(17, 22)):
    best_gdf = None
    best_epsg = None
    best_span = float("inf")

    for z in zones:
        epsg = 32700 + z
        try:
            g_try = gdf.copy()
            g_try = g_try.set_crs(epsg, allow_override=True)
            g_wgs = g_try.to_crs(4326)
            bounds = g_wgs.total_bounds

            if bounds_reasonable(bounds, max_span_deg=10):
                span = abs(bounds[2] - bounds[0]) + abs(bounds[3] - bounds[1])
                if span < best_span:
                    best_span = span
                    best_epsg = epsg
                    best_gdf = g_try
        except Exception:
            continue

    return best_gdf, best_epsg


def coerce_crs_safely(gdf: Optional[gpd.GeoDataFrame]):
    if gdf is None or len(gdf) == 0:
        return gdf

    gdf = gdf.copy()

    try:
        bounds = gdf.total_bounds
    except Exception:
        return gdf

    if gdf.crs is None:
        if detect_longlat_bounds(bounds):
            return gdf.set_crs(4326)

        guessed, _ = guess_peru_utm(gdf)
        if guessed is not None:
            return guessed

        return gdf.set_crs(CRS_METRICO, allow_override=True)

    try:
        epsg_now = gdf.crs.to_epsg()
    except Exception:
        epsg_now = None

    if epsg_now == 4326 and not detect_longlat_bounds(bounds):
        g_no = gdf.copy()
        g_no = g_no.set_crs(None, allow_override=True)

        guessed, _ = guess_peru_utm(g_no)
        if guessed is not None:
            return guessed

        return gdf.set_crs(CRS_METRICO, allow_override=True)

    return gdf


def to_wgs(gdf: Optional[gpd.GeoDataFrame]):
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = coerce_crs_safely(gdf)
    epsg = gdf.crs.to_epsg() if gdf.crs else None
    if epsg != CRS_WGS:
        gdf = gdf.to_crs(CRS_WGS)
    return gdf


def to_metric(gdf: Optional[gpd.GeoDataFrame], target_epsg: Optional[int] = None):
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = coerce_crs_safely(gdf)
    epsg_target = target_epsg if target_epsg is not None else metric_crs_for_gdf(gdf)
    epsg_now = gdf.crs.to_epsg() if gdf.crs else None
    if epsg_now != epsg_target:
        gdf = gdf.to_crs(epsg_target)
    return gdf


def safe_read_gdf(path: Path):
    if not path.exists():
        return None
    gdf = gpd.read_file(str(path))
    gdf = coerce_crs_safely(gdf)
    gdf = to_wgs(gdf)
    return gdf


def hydro_year_start(d: pd.Timestamp) -> pd.Timestamp:
    year = d.year
    return pd.Timestamp(year=year if d.month >= 9 else year - 1, month=9, day=1)



def map_to_ref_dates(dates: pd.Series, ref_year: int = REF_YEAR) -> pd.Series:
    out = []
    for d in pd.to_datetime(dates, errors="coerce"):
        if pd.isna(d):
            out.append(pd.NaT)
            continue

        yy = ref_year if d.month >= 9 else ref_year + 1

        try:
            out.append(pd.Timestamp(year=yy, month=d.month, day=d.day))
        except Exception:
            out.append(pd.NaT)

    return pd.Series(out, index=dates.index)

def safe_key_piece(x):
    if x is None or pd.isna(x):
        return "na"
    return str(x).replace(" ", "_").replace("/", "_")


def fit_map_to_gdf(
    m: folium.Map,
    gdf: Optional[gpd.GeoDataFrame],
    fallback_gdf: Optional[gpd.GeoDataFrame] = None,
):
    def _fit(candidate):
        if candidate is None or len(candidate) == 0:
            return False

        candidate = to_wgs(candidate)

        try:
            geom_types = set(candidate.geometry.geom_type.dropna().astype(str).tolist())
        except Exception:
            return False

        try:
            if geom_types.issubset({"Point", "MultiPoint"}):
                geom = candidate.geometry.iloc[0]
                if geom.geom_type == "Point":
                    m.location = [geom.y, geom.x]
                    m.zoom_start = 14
                    return True
                elif geom.geom_type == "MultiPoint" and len(geom.geoms) > 0:
                    pt = list(geom.geoms)[0]
                    m.location = [pt.y, pt.x]
                    m.zoom_start = 14
                    return True
        except Exception:
            pass

        try:
            bounds = candidate.total_bounds
        except Exception:
            return False

        if not detect_longlat_bounds(bounds):
            return False

        xmin, ymin, xmax, ymax = bounds

        if abs(xmax - xmin) > 20 or abs(ymax - ymin) > 20:
            return False

        if abs(xmax - xmin) < 0.0001 and abs(ymax - ymin) < 0.0001:
            dx = 0.01
            dy = 0.01
            xmin, xmax = xmin - dx, xmax + dx
            ymin, ymax = ymin - dy, ymax + dy

        m.fit_bounds([[ymin, xmin], [ymax, xmax]])
        return True

    if _fit(gdf):
        return
    if _fit(fallback_gdf):
        return

    m.location = [AOI_LAT, AOI_LON]
    m.zoom_start = AOI_ZOOM

def make_folium_map(tiles="OpenStreetMap"):
    return folium.Map(
        location=[AOI_LAT, AOI_LON],
        zoom_start=AOI_ZOOM,
        tiles=tiles,
        control_scale=True
    )



def add_gdf_to_map(
    m: folium.Map,
    gdf: Optional[gpd.GeoDataFrame],
    color: str,
    fill: bool = True,
    weight: int = 2,
    tooltip_fields: Optional[list] = None,
):
    if gdf is None or len(gdf) == 0:
        return

    gdf = to_wgs(gdf)

    try:
        geom_types = set(gdf.geometry.geom_type.dropna().astype(str).tolist())
    except Exception:
        geom_types = set()

    if geom_types.issubset({"Point", "MultiPoint"}):
        fields_ok = [c for c in (tooltip_fields or []) if c in gdf.columns]

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            pts = [geom] if geom.geom_type == "Point" else list(geom.geoms)

            tt = None
            if fields_ok:
                tt = "<br>".join([f"<b>{f}</b>: {row[f]}" for f in fields_ok])

            for pt in pts:
                folium.CircleMarker(
                    location=[pt.y, pt.x],
                    radius=7,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=1.0,
                    tooltip=tt,
                ).add_to(m)
        return

    tooltip = None
    if tooltip_fields:
        fields_ok = [c for c in tooltip_fields if c in gdf.columns]
        if fields_ok:
            tooltip = folium.GeoJsonTooltip(fields=fields_ok)

    def style_fn(_):
        return {
            "color": color,
            "fillColor": color,
            "weight": weight,
            "fillOpacity": 0.55 if fill else 0.0,
        }

    folium.GeoJson(
        data=gdf.__geo_interface__,
        style_function=style_fn,
        tooltip=tooltip
    ).add_to(m)

def add_map_legend(m: folium.Map, title: str, items: list, position: str = "topright"):
    pos_map = {
        "topright": "top:12px; right:12px;",
        "topleft": "top:12px; left:12px;"
    }
    pos_css = pos_map.get(position, "top:12px; right:12px;")

    rows = []
    for label, color, style in items:
        if style == "line":
            sw = f'''
            <span style="
                display:inline-block;
                width:24px;
                border-top:4px solid {color};
                margin-right:8px;
                vertical-align:middle;
            "></span>
            '''
        else:
            sw = f'''
            <span style="
                display:inline-block;
                width:16px;
                height:16px;
                border-radius:50%;
                background:{color};
                border:1px solid rgba(0,0,0,.25);
                margin-right:8px;
                vertical-align:middle;
            "></span>
            '''
        rows.append(f'<div style="margin:8px 0; font-size:13px; white-space:nowrap;">{sw}{label}</div>')

    html = f"""
    <div style="
        position:absolute;
        z-index:9999;
        {pos_css}
        background:white;
        padding:10px 12px;
        border-radius:12px;
        min-width:210px;
        box-shadow:0 4px 14px rgba(0,0,0,.18);
        border:1px solid #d9dee7;
    ">
      <div style="font-weight:800; font-size:15px; margin-bottom:6px;">{title}</div>
      <div style="border-top:1px solid #e5e7eb; padding-top:6px;">
        {''.join(rows)}
      </div>
    </div>
    """
    m.get_root().html.add_child(Element(html))


def nivel_card_html(level_txt: str, level_color: str, q_val: float) -> str:
    q_txt = f"{q_val:.2f}" if pd.notna(q_val) else "SIN DATO"
    return f"""
    <div class="pbi-right-card">
        <div class="pbi-right-title">NIVEL</div>
        <div class="nivel-box" style="background:{level_color}; margin-bottom:8px;">{level_txt}</div>
        <div style="text-align:center; font-weight:800; color:#5b6778;">
            Caudal: {q_txt} m³/s
        </div>
    </div>
    """


def exp_cards_html(df: pd.DataFrame) -> str:
    icon_map = {
        "Población": "👥",
        "Instituciones educativas": "🏫",
        "Centros de salud": "🏥",
        "Red vial": "🚚",
        "Áreas agrícolas": "🌿"
    }

    unit_map = {
        "Población": "[hab.]",
        "Instituciones educativas": "[und.]",
        "Centros de salud": "[und.]",
        "Red vial": "[km.]",
        "Áreas agrícolas": "[ha.]"
    }

    def fmt_value(elem, v):
        if pd.isna(v):
            v = 0
        if elem in ["Red vial", "Áreas agrícolas"]:
            return f"{float(v):.2f}"
        if elem == "Población":
            return f"{int(round(v)):,}"
        return str(int(round(v)))

    rows = []
    for _, row in df.iterrows():
        elem = row["Elemento"]
        val = fmt_value(elem, row["Valor"])
        ico = icon_map.get(elem, "•")
        unit = unit_map.get(elem, "")

        rows.append(f"""
        <div class="exp-row">
            <div class="exp-left">
                <div class="exp-ico">{ico}</div>
                <div class="exp-txt">
                    <div class="exp-name">{elem}</div>
                    <div class="exp-unit">{unit}</div>
                </div>
            </div>
            <div class="exp-right">{val}</div>
        </div>
        """)

    return f"""
    <div class="pbi-right-card">
        <div class="section-title" style="margin-top:0;">ELEMENTOS EXPUESTOS</div>
        <div class="exp-grid">
            {''.join(rows)}
        </div>
    </div>
    """


# ============================================================
# GOOGLE DRIVE
# ============================================================
@st.cache_resource
def get_drive_service():
    # 1) Prioridad: Secrets de Streamlit Cloud
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build("drive", "v3", credentials=creds)

    # 2) Fallback local: JSON en tu PC
    if JSON_GOOGLE.exists():
        creds = service_account.Credentials.from_service_account_file(
            str(JSON_GOOGLE),
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build("drive", "v3", credentials=creds)

    raise FileNotFoundError(
        "No se encontraron credenciales de Google. "
        "En Streamlit Cloud usa st.secrets; localmente puedes usar google-service.json."
    )


def download_drive_file(service, folder_id: str, filename: str, dest: Path):
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    files = resp.get("files", [])

    if not files:
        raise FileNotFoundError(f"No se encontró en Drive: {filename}")

    file_id = files[0]["id"]
    request = service.files().get_media(fileId=file_id)

    with io.BytesIO() as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        dest.write_bytes(fh.getvalue())


@st.cache_data(show_spinner=True)
def ensure_remote_files():
    try:
        service = get_drive_service()
        download_drive_file(service, DRIVE_FOLDER_ID, "hist_filtrado.parquet", HIST_PATH)
        download_drive_file(service, DRIVE_FOLDER_ID, "fore_filtrado.parquet", FORE_PATH)
    except Exception as e:
        if not HIST_PATH.exists() or not FORE_PATH.exists():
            raise RuntimeError(
                f"No se pudo descargar desde Drive y no existen parquet locales. Detalle: {e}"
            )
    return str(HIST_PATH), str(FORE_PATH)


# ============================================================
# CARGA DE DATOS
# ============================================================
@st.cache_data(show_spinner=True)
def load_parquet_data():
    ensure_remote_files()

    hist = pd.read_parquet(HIST_PATH)
    fore = pd.read_parquet(FORE_PATH)

    hist["fecha"] = pd.to_datetime(hist["fecha"]).dt.normalize()
    fore["fecha"] = pd.to_datetime(fore["fecha"]).dt.normalize()

    hist["comid"] = pd.to_numeric(hist["comid"], errors="coerce")
    fore["comid"] = pd.to_numeric(fore["comid"], errors="coerce")

    hist = hist.dropna(subset=["fecha", "comid"]).copy()
    fore = fore.dropna(subset=["fecha", "comid"]).copy()

    comids = sorted(pd.unique(pd.concat([hist["comid"], fore["comid"]]).dropna()).tolist())
    return hist, fore, comids



@st.cache_data(show_spinner=True)
def load_estaciones():
    if not SHP_ESTACIONES.exists():
        raise FileNotFoundError(f"No existe el shapefile de estaciones: {SHP_ESTACIONES}")

    gdf = gpd.read_file(str(SHP_ESTACIONES))
    gdf = coerce_crs_safely(gdf)
    gdf = to_wgs(gdf)

    if "COMID" not in gdf.columns:
        raise ValueError("El shapefile de estaciones debe tener la columna COMID.")

    if "Estacion" not in gdf.columns and "ESTACION" in gdf.columns:
        gdf = gdf.rename(columns={"ESTACION": "Estacion"})

    if "Estacion" not in gdf.columns:
        gdf["Estacion"] = gdf["COMID"].astype(str)

    col_zona = find_col(gdf.columns, ["ZONA_UTM", "zona_utm", "zona", "utm"])
    if col_zona is not None:
        gdf["ZONA_UTM"] = gdf[col_zona].astype(str).str.strip()
        gdf["ZONA_FOLDER"] = gdf["ZONA_UTM"].apply(normalize_zone_value)
    else:
        gdf["ZONA_UTM"] = None
        gdf["ZONA_FOLDER"] = gdf.geometry.x.apply(zone_folder_from_lon)

    gdf["ZONA_FOLDER"] = gdf["ZONA_FOLDER"].fillna("utm19s")
    return gdf


@st.cache_data(show_spinner=True)
def load_base_layers(zone_key: str):
    zone_dir = zone_dir_from_key(zone_key)

    file_agricola = zone_dir / "Agricola.shp"
    file_poblacion = zone_dir / "Poblacion.shp"
    file_vial = zone_dir / "Red Vial.shp"
    file_edu = zone_dir / "Instituciones Educativas.shp"
    file_salud = zone_dir / "Puesto de Salud.shp"

    agri = safe_read_gdf(file_agricola)
    pop = safe_read_gdf(file_poblacion)
    vial = safe_read_gdf(file_vial)
    edu = safe_read_gdf(file_edu)
    salud = safe_read_gdf(file_salud)

    if pop is not None:
        colp = find_col(pop.columns, [
            "poblacion", "población", "hab", "habitantes", "pob", "pop", "pob_total", "poblac"
        ])
        if colp is None:
            pop["__pob__"] = 1
        else:
            pop["__pob__"] = pd.to_numeric(pop[colp], errors="coerce").fillna(0)

    return {
        "agri_wgs": agri,
        "pop_wgs": pop,
        "vial_wgs": vial,
        "edu_wgs": edu,
        "salud_wgs": salud,
        "zone_key": zone_key,
    }


@st.cache_data(show_spinner=True)
def load_flood_index(zone_key: str):
    zone_dir = zone_dir_from_key(zone_key)
    file_inundacion = zone_dir / inundacion_filename_for_zone(zone_key)

    if not file_inundacion.exists():
        return gpd.GeoDataFrame(
            columns=["Distrito", "Caudal", "COMID", "ZONA_UTM", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )

    g = gpd.read_file(str(file_inundacion))
    g = coerce_crs_safely(g)
    g = to_wgs(g)

    col_dist = find_col(g.columns, ["Distrito", "DISTRITO", "Dist", "NOMBDIST", "NOM_DIST"])
    col_q = find_col(g.columns, ["Caudal", "CAUDAL", "Q", "QMAX", "Qmax", "flow", "discharge"])
    col_comid = find_col(g.columns, ["COMID", "comid"])
    col_zona = find_col(g.columns, ["ZONA_UTM", "zona_utm", "zona", "utm"])

    if col_dist is None or col_q is None or col_comid is None:
        raise ValueError("El shape de inundación debe tener columnas COMID, Distrito y Caudal.")

    keep_cols = [col_dist, col_q, col_comid, "geometry"]
    rename_map = {
        col_dist: "Distrito",
        col_q: "Caudal",
        col_comid: "COMID",
    }

    if col_zona is not None:
        keep_cols.insert(3, col_zona)
        rename_map[col_zona] = "ZONA_UTM"

    tmp = g[keep_cols].copy().rename(columns=rename_map)
    tmp["Distrito"] = tmp["Distrito"].astype(str).str.strip()
    tmp["Caudal"] = pd.to_numeric(tmp["Caudal"], errors="coerce")
    tmp["COMID"] = pd.to_numeric(tmp["COMID"], errors="coerce")

    if "ZONA_UTM" not in tmp.columns:
        tmp["ZONA_UTM"] = zone_key.replace("utm", "").replace("s", "")

    tmp = tmp.dropna(subset=["Distrito", "Caudal", "COMID", "geometry"])
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs="EPSG:4326")
    return tmp

# ============================================================
# LÓGICA DE NEGOCIO
# ============================================================
def get_station_series(hist: pd.DataFrame, fore: pd.DataFrame, comid_sel: float):
    h = hist[hist["comid"] == comid_sel].copy().sort_values("fecha")
    f = fore[fore["comid"] == comid_sel].copy().sort_values("fecha")
    return h, f


def q_now(hist_station: pd.DataFrame):
    if hist_station.empty:
        return float("nan")
    return float(hist_station["qr_hist"].iloc[-1])


def q_gfs_day(hist_station: pd.DataFrame, fore_station: pd.DataFrame, nday: int):
    if hist_station.empty or fore_station.empty:
        return float("nan")

    last_hist_date = hist_station["fecha"].max()

    daily = (
        fore_station.groupby("fecha", as_index=False)["qr_gfs"]
        .first()
        .sort_values("fecha")
    )

    daily = daily[daily["fecha"] > last_hist_date].copy()

    if len(daily) < nday:
        return float("nan")

    return float(daily.iloc[nday - 1]["qr_gfs"])


def distritos_por_comid(flood_index: gpd.GeoDataFrame, comid_sel: float):
    if flood_index is None or len(flood_index) == 0 or pd.isna(comid_sel):
        return []
    sub = flood_index[pd.to_numeric(flood_index["COMID"], errors="coerce") == float(comid_sel)].copy()
    if len(sub) == 0:
        return []
    return sorted(sub["Distrito"].dropna().astype(str).unique().tolist())


def nivel_from_q(q: float):
    if pd.isna(q):
        return "SIN DATO", "#94a3b8"
    if q >= 400:
        return "EXTREMO", "#c62828"
    if q >= 320:
        return "ALTO", "#ef6c00"
    if q >= 260:
        return "MODERADO", "#f9a825"
    return "BAJO", "#2e7d32"



def flood_geom_from_qd(flood_index: gpd.GeoDataFrame, q: float, distrito: str, comid_sel: float):
    if flood_index is None or len(flood_index) == 0:
        return None

    if pd.isna(q) or pd.isna(comid_sel) or not distrito:
        return None

    sub = flood_index[
        (pd.to_numeric(flood_index["COMID"], errors="coerce") == float(comid_sel)) &
        (flood_index["Distrito"].astype(str).str.strip() == str(distrito).strip())
    ].copy()

    if len(sub) == 0:
        return None

    sub["Caudal"] = pd.to_numeric(sub["Caudal"], errors="coerce")
    sub = sub.dropna(subset=["Caudal"])

    if len(sub) == 0:
        return None

    sub["dist_q"] = (sub["Caudal"] - float(q)).abs()
    sub = sub.sort_values("dist_q")
    return sub.iloc[[0]].drop(columns=["dist_q"])

def inters_polys(base_wgs: Optional[gpd.GeoDataFrame], flood_wgs: Optional[gpd.GeoDataFrame], target_epsg: Optional[int] = None):
    if base_wgs is None or flood_wgs is None or len(base_wgs) == 0 or len(flood_wgs) == 0:
        return None

    base_utm = to_metric(base_wgs, target_epsg=target_epsg)
    flood_utm = to_metric(flood_wgs, target_epsg=target_epsg)
    try:
        return gpd.overlay(base_utm, flood_utm, how="intersection", keep_geom_type=False)
    except Exception:
        return None


def vial_touch(vial_wgs: Optional[gpd.GeoDataFrame], flood_wgs: Optional[gpd.GeoDataFrame], target_epsg: Optional[int] = None):
    if vial_wgs is None or flood_wgs is None or len(vial_wgs) == 0 or len(flood_wgs) == 0:
        return None

    vial_utm = to_metric(vial_wgs, target_epsg=target_epsg)
    flood_utm = to_metric(flood_wgs, target_epsg=target_epsg)
    try:
        return gpd.clip(vial_utm, flood_utm)
    except Exception:
        return None


def compute_exposures(base_layers: dict, flood_wgs: Optional[gpd.GeoDataFrame]):
    target_epsg = metric_crs_for_gdf(flood_wgs) if flood_wgs is not None and len(flood_wgs) else CRS_METRICO

    return {
        "agri_af": inters_polys(base_layers["agri_wgs"], flood_wgs, target_epsg=target_epsg),
        "pop_af": inters_polys(base_layers["pop_wgs"], flood_wgs, target_epsg=target_epsg),
        "edu_af": inters_polys(base_layers["edu_wgs"], flood_wgs, target_epsg=target_epsg),
        "salud_af": inters_polys(base_layers["salud_wgs"], flood_wgs, target_epsg=target_epsg),
        "vial_af": vial_touch(base_layers["vial_wgs"], flood_wgs, target_epsg=target_epsg),
    }


def make_exp_df(pop_af, edu_af, salud_af, vial_af, agri_af):
    pop_n = 0 if pop_af is None or len(pop_af) == 0 else float(pop_af["__pob__"].fillna(0).sum())
    edu_n = 0 if edu_af is None else int(len(edu_af))
    salud_n = 0 if salud_af is None else int(len(salud_af))
    vial_km = 0 if vial_af is None or len(vial_af) == 0 else float(vial_af.length.sum() / 1000)
    agri_ha = 0 if agri_af is None or len(agri_af) == 0 else float(agri_af.area.sum() / 10000)

    return pd.DataFrame({
        "Elemento": [
            "Población",
            "Instituciones educativas",
            "Centros de salud",
            "Red vial",
            "Áreas agrícolas"
        ],
        "Valor": [pop_n, edu_n, salud_n, vial_km, agri_ha],
        "Unidad": ["hab.", "und.", "und.", "km", "ha"]
    })


# ============================================================
# GRÁFICOS
# ============================================================

def make_hist_chart(h: pd.DataFrame):
    ult = h["fecha"].max()
    ini_hid = hydro_year_start(ult)

    clima = h.copy()
    clima = clima.dropna(subset=["fecha", "qr_hist"]).copy()
    clima["mmdd"] = clima["fecha"].dt.strftime("%m-%d")
    clim = clima.groupby("mmdd", as_index=False)["qr_hist"].mean()

    def mmdd_to_refdate(mmdd: str):
        try:
            mm = int(mmdd[:2])
            dd = int(mmdd[3:5])
            yy = REF_YEAR if mm >= 9 else REF_YEAR + 1
            return pd.Timestamp(year=yy, month=mm, day=dd)
        except Exception:
            return pd.NaT

    clim["fecha_ref"] = clim["mmdd"].apply(mmdd_to_refdate)
    clim = clim.dropna(subset=["fecha_ref"]).sort_values("fecha_ref")

    curr = h[h["fecha"] >= ini_hid].copy()
    curr = curr.dropna(subset=["fecha", "qr_hist"]).copy()
    curr["fecha_ref"] = map_to_ref_dates(curr["fecha"], ref_year=REF_YEAR)
    curr = curr.dropna(subset=["fecha_ref"]).sort_values("fecha_ref")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clim["fecha_ref"],
        y=clim["qr_hist"],
        mode="lines",
        name="Climatología diaria",
        line=dict(width=2, color="#1f77b4")
    ))
    fig.add_trace(go.Scatter(
        x=curr["fecha_ref"],
        y=curr["qr_hist"],
        mode="lines",
        name="Año hidrológico actual",
        line=dict(width=2, color="black")
    ))

    fig.update_layout(
        title="Caudal medio diario simulado",
        xaxis_title="",
        yaxis_title="Caudal [m³/s]",
        hovermode="x unified",
        height=320,
        legend=dict(orientation="h")
    )
    return fig

def make_fore_chart(h: pd.DataFrame, f: pd.DataFrame):
    t_hist = h.tail(14)[["fecha", "qr_hist"]].copy()
    last_q = t_hist["qr_hist"].iloc[-1]
    last_t = t_hist["fecha"].iloc[-1]

    daily = f.groupby("fecha", as_index=False)[["qr_eta_eqm", "qr_eta_scal", "qr_gfs", "qr_wrf"]].first()
    daily = daily[daily["fecha"] >= last_t].copy().sort_values("fecha")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_hist["fecha"], y=t_hist["qr_hist"],
        mode="lines+markers", name="Qsim histórico",
        line=dict(color="gray")
    ))
    fig.add_trace(go.Scatter(
        x=[last_t] + daily["fecha"].tolist(),
        y=[last_q] + daily["qr_eta_eqm"].tolist(),
        mode="lines+markers", name="ETA_eqm"
    ))
    fig.add_trace(go.Scatter(
        x=[last_t] + daily["fecha"].tolist(),
        y=[last_q] + daily["qr_eta_scal"].tolist(),
        mode="lines+markers", name="ETA_scal"
    ))
    fig.add_trace(go.Scatter(
        x=[last_t] + daily["fecha"].tolist(),
        y=[last_q] + daily["qr_gfs"].tolist(),
        mode="lines+markers", name="GFS"
    ))
    fig.add_trace(go.Scatter(
        x=[last_t] + daily["fecha"].tolist(),
        y=[last_q] + daily["qr_wrf"].tolist(),
        mode="lines+markers", name="WRF"
    ))

    fig.update_layout(
        title="Pronóstico",
        xaxis_title="Fecha",
        yaxis_title="Caudal [m³/s]",
        hovermode="x unified",
        height=320,
        legend=dict(orientation="h")
    )
    return fig


def make_var_chart(h: pd.DataFrame, f: pd.DataFrame):
    q_last = float(h["qr_hist"].iloc[-1])

    daily = f.groupby("fecha", as_index=False)[["qr_eta_eqm", "qr_eta_scal", "qr_gfs"]].first()
    daily = daily.sort_values("fecha").head(3).copy()

    daily["ETA_eqm_var"] = ((daily["qr_eta_eqm"] - q_last) / q_last) * 100
    daily["ETA_scal_var"] = ((daily["qr_eta_scal"] - q_last) / q_last) * 100
    daily["GFS_var"] = ((daily["qr_gfs"] - q_last) / q_last) * 100

    fig = go.Figure()
    fig.add_bar(x=daily["fecha"], y=daily["ETA_eqm_var"], name="ETA_eqm")
    fig.add_bar(x=daily["fecha"], y=daily["ETA_scal_var"], name="ETA_scal")
    fig.add_bar(x=daily["fecha"], y=daily["GFS_var"], name="GFS")

    fig.update_layout(
        title="Variación pronosticada (3 días)",
        xaxis_title="Fecha",
        yaxis_title="Variación [%]",
        hovermode="x unified",
        barmode="group",
        height=300,
        legend=dict(orientation="h")
    )
    return fig


# ============================================================
# BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown("### Opciones")
    if st.button("Recargar caché y datos"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("### Rutas")
    st.caption(f"Base: {BASE_DIR}")
    st.caption(f"Shapes: {SHAPE_BASE}")
    if "comid_sel" in st.session_state:
        st.caption(f"Zona activa: {zone_folder_for_comid(estaciones_validas, float(st.session_state.comid_sel)) if 'estaciones_validas' in locals() else '(pendiente)'}")


# ============================================================
# CARGA MAESTRA
# ============================================================
try:
    hist, fore, comids = load_parquet_data()
    estaciones = load_estaciones()
except Exception as e:
    st.error(f"Error cargando la aplicación: {e}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================
estaciones_validas = estaciones[estaciones["COMID"].astype(float).isin([float(c) for c in comids])].copy()
if estaciones_validas.empty:
    st.error("No hay COMID comunes entre estaciones y parquet.")
    st.stop()

if "comid_sel" not in st.session_state:
    st.session_state.comid_sel = float(estaciones_validas["COMID"].astype(float).iloc[0])

if "select_estacion_comid" in st.session_state:
    try:
        st.session_state.comid_sel = float(st.session_state["select_estacion_comid"])
    except Exception:
        pass

for key in ["dist_now", "dist_d1", "dist_d2", "dist_d3"]:
    if key not in st.session_state:
        st.session_state[key] = None

zone_key_sel = zone_folder_for_comid(estaciones_validas, float(st.session_state.comid_sel))
if zone_key_sel is None:
    zone_key_sel = "utm19s"

try:
    base_layers = load_base_layers(zone_key_sel)
    flood_index = load_flood_index(zone_key_sel)
except Exception as e:
    st.error(f"Error cargando shapes para la zona {zone_key_sel}: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="titulo-app">PRONÓSTICOS BASADOS EN IMPACTO DEL RÍO VILCANOTA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitulo-app">Versión en Streamlit basada en la lógica de tu app Shiny</div>', unsafe_allow_html=True)

tab_pron, tab_pbi = st.tabs(["PRONÓSTICO", "PBI"])


# ============================================================
# TAB PRONÓSTICO
# ============================================================
with tab_pron:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        opciones = estaciones_validas["COMID"].astype(float).tolist()
        etiquetas = {
            float(row["COMID"]): f"{row['Estacion']} | COMID {row['COMID']}"
            for _, row in estaciones_validas.iterrows()
        }

        comid_prev = st.session_state.get("comid_sel")
        st.session_state.comid_sel = st.selectbox(
            "Estación / COMID",
            options=opciones,
            index=opciones.index(st.session_state.comid_sel) if st.session_state.comid_sel in opciones else 0,
            format_func=lambda x: etiquetas.get(float(x), str(x)),
            key="select_estacion_comid"
        )

        if comid_prev != st.session_state.comid_sel:
            for _k in ["dist_now", "dist_d1", "dist_d2", "dist_d3"]:
                st.session_state[_k] = None
            st.rerun()

        est_sel = estaciones_validas[
            estaciones_validas["COMID"].astype(float) == float(st.session_state.comid_sel)
        ].copy()

        m_est = make_folium_map(tiles="OpenStreetMap")
        add_gdf_to_map(
            m_est,
            est_sel,
            "#1E90FF",
            fill=True,
            weight=3,
            tooltip_fields=["Estacion", "COMID"]
        )
        fit_map_to_gdf(m_est, est_sel)

        st_folium(
            m_est,
            width=None,
            height=520,
            returned_objects=[],
            key="map_pron_estaciones"
        )

    with col_right:
        h, f = get_station_series(hist, fore, float(st.session_state.comid_sel))

        if h.empty or f.empty:
            st.warning("No hay datos suficientes para la estación seleccionada.")
        else:
            st.plotly_chart(make_hist_chart(h), use_container_width=True)
            st.plotly_chart(make_fore_chart(h, f), use_container_width=True)
            st.plotly_chart(make_var_chart(h, f), use_container_width=True)
            st.caption("Climatología: promedio por día del año. Año hidrológico: septiembre a agosto.")

# ============================================================
# FUNCIÓN REUTILIZABLE PBI
# ============================================================
def render_pbi_panel(panel_id: str, q_val: float, fecha_texto: str):
    st.markdown(f'<div class="fecha-box">{fecha_texto}</div>', unsafe_allow_html=True)

    station_gdf = estaciones_validas[
        estaciones_validas["COMID"].astype(float) == float(st.session_state.comid_sel)
    ].copy()

    c_map1, c_map2, c_side = st.columns([1.2, 1.2, 0.7])

    distritos = distritos_por_comid(flood_index, float(st.session_state.comid_sel))
    distrito_sel = st.session_state.get(f"dist_{panel_id}", distritos[0] if distritos else None)

    with c_side:
        level_txt, level_color = nivel_from_q(q_val)
        st.html(nivel_card_html(level_txt, level_color, q_val))

        with st.container(border=True):
            st.markdown('<div class="pbi-right-title">DISTRITO</div>', unsafe_allow_html=True)
            if distritos:
                distrito_sel = st.selectbox(
                    "Distrito",
                    options=distritos,
                    index=distritos.index(distrito_sel) if distrito_sel in distritos else 0,
                    key=f"selector_dist_{panel_id}",
                    label_visibility="collapsed"
                )
                st.session_state[f"dist_{panel_id}"] = distrito_sel
            else:
                st.warning(f"No se detectaron distritos para el COMID seleccionado en la zona {base_layers.get('zone_key', '')}.")
                distrito_sel = None

    flood_gdf = flood_geom_from_qd(flood_index, q_val, distrito_sel, float(st.session_state.comid_sel)) if distrito_sel else None
    exp = compute_exposures(base_layers, flood_gdf)

    q_key = safe_key_piece(round(q_val, 2) if pd.notna(q_val) else "na")
    dist_key = safe_key_piece(distrito_sel)

    with c_map1:
        st.markdown('<div class="section-title">MAPA DE INUNDACIÓN</div>', unsafe_allow_html=True)

        m1 = make_folium_map(tiles="OpenStreetMap")
        add_gdf_to_map(m1, flood_gdf, "#1E90FF", fill=True, weight=3)
        fit_map_to_gdf(m1, flood_gdf, fallback_gdf=station_gdf)

        add_map_legend(
            m1,
            "Capas",
            [
                ("Inundación", "#1E90FF", "fill"),
            ],
            position="topright"
        )

        st_folium(
            m1,
            width=None,
            height=560,
            returned_objects=[],
            key=f"map_inund_{panel_id}_{dist_key}_{q_key}"
        )

    with c_map2:
        st.markdown('<div class="section-title">AFECTACIONES Y ELEMENTOS EXPUESTOS</div>', unsafe_allow_html=True)

        m2 = make_folium_map(tiles="OpenStreetMap")

        geom_fit = None

        if exp["agri_af"] is not None and len(exp["agri_af"]):
            add_gdf_to_map(m2, exp["agri_af"], "#00B050", fill=True, weight=2)
            geom_fit = exp["agri_af"]

        if exp["pop_af"] is not None and len(exp["pop_af"]):
            add_gdf_to_map(m2, exp["pop_af"], "#b71c1c", fill=True, weight=2, tooltip_fields=["__pob__"])
            if geom_fit is None:
                geom_fit = exp["pop_af"]

        if exp["edu_af"] is not None and len(exp["edu_af"]):
            add_gdf_to_map(m2, exp["edu_af"], "#0d47a1", fill=True, weight=2)
            if geom_fit is None:
                geom_fit = exp["edu_af"]

        if exp["salud_af"] is not None and len(exp["salud_af"]):
            add_gdf_to_map(m2, exp["salud_af"], "#c62828", fill=True, weight=2)
            if geom_fit is None:
                geom_fit = exp["salud_af"]

        if exp["vial_af"] is not None and len(exp["vial_af"]):
            add_gdf_to_map(m2, exp["vial_af"], "#FFFFFF", fill=False, weight=8)
            add_gdf_to_map(m2, exp["vial_af"], "#8E24AA", fill=False, weight=5)
            if geom_fit is None:
                geom_fit = exp["vial_af"]

        if geom_fit is None:
            geom_fit = flood_gdf

        fit_map_to_gdf(m2, geom_fit, fallback_gdf=station_gdf)

        add_map_legend(
            m2,
            "Capas",
            [
                ("Agrícola afectada", "#00B050", "fill"),
                ("Población afectada", "#b71c1c", "fill"),
                ("Vial afectada", "#8E24AA", "line"),
                ("Educación afectada", "#0d47a1", "fill"),
                ("Salud afectada", "#c62828", "fill"),
            ],
            position="topright"
        )

        st_folium(
            m2,
            width=None,
            height=560,
            returned_objects=[],
            key=f"map_afecta_{panel_id}_{dist_key}_{q_key}"
        )

    exp_df = make_exp_df(
        exp["pop_af"], exp["edu_af"], exp["salud_af"], exp["vial_af"], exp["agri_af"]
    ).copy()

    with c_side:
        st.html(exp_cards_html(exp_df))


# ============================================================
# TAB PBI
# ============================================================
with tab_pbi:
    h, f = get_station_series(hist, fore, float(st.session_state.comid_sel))

    if h.empty:
        st.warning("No hay datos históricos para la estación seleccionada.")
    else:
        ult_fecha = h["fecha"].max()

        daily_gfs = (
            f.groupby("fecha", as_index=False)["qr_gfs"]
            .first()
            .sort_values("fecha")
        )
        daily_gfs = daily_gfs[daily_gfs["fecha"] > ult_fecha].copy()
        fechas_gfs = daily_gfs["fecha"].tolist()

        sub_now, sub_d1, sub_d2, sub_d3 = st.tabs([
            "Estado actual",
            "Pronóstico día 1",
            "Pronóstico día 2",
            "Pronóstico día 3"
        ])

        with sub_now:
            render_pbi_panel(
                panel_id="now",
                q_val=q_now(h),
                fecha_texto=f"FECHA ACTUAL: {ult_fecha.strftime('%d/%m/%Y')}"
            )

        with sub_d1:
            fecha_txt = fechas_gfs[0].strftime("%d/%m/%Y") if len(fechas_gfs) >= 1 else "SIN DATO"
            render_pbi_panel(
                panel_id="d1",
                q_val=q_gfs_day(h, f, 1),
                fecha_texto=f"FECHA DE PRONÓSTICO: {fecha_txt}"
            )

        with sub_d2:
            fecha_txt = fechas_gfs[1].strftime("%d/%m/%Y") if len(fechas_gfs) >= 2 else "SIN DATO"
            render_pbi_panel(
                panel_id="d2",
                q_val=q_gfs_day(h, f, 2),
                fecha_texto=f"FECHA DE PRONÓSTICO: {fecha_txt}"
            )

        with sub_d3:
            fecha_txt = fechas_gfs[2].strftime("%d/%m/%Y") if len(fechas_gfs) >= 3 else "SIN DATO"
            render_pbi_panel(
                panel_id="d3",
                q_val=q_gfs_day(h, f, 3),
                fecha_texto=f"FECHA DE PRONÓSTICO: {fecha_txt}"
            )
