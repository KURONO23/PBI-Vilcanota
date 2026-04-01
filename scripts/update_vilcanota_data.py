from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime
from ftplib import FTP
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile  # pyshp
import xarray as xr
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


# ============================================================
# TÍTULO: Lectura de NetCDF desde FTP en memoria y reemplazo de archivos en Google Drive
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SHP_PATH = Path(os.getenv("SHP_PATH", BASE_DIR / "shape" / "Estacion.shp"))

FTP_HOST = os.getenv("FTP_HOST", "").strip()
FTP_USER = os.getenv("FTP_USER", "").strip()
FTP_PASS = os.getenv("FTP_PASS", "").strip()
FTP_DIR = os.getenv("FTP_DIR", "/").strip()

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "").strip()
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON", "").strip()
GOOGLE_SERVICE_JSON_PATH = os.getenv("GOOGLE_SERVICE_JSON_PATH", "").strip()

HIST_NAME = "hist_filtrado.parquet"
FORE_NAME = "fore_filtrado.parquet"
META_NAME = "meta_filtrado.json"


def require_env() -> None:
    faltan = []
    if not FTP_HOST:
        faltan.append("FTP_HOST")
    if not FTP_USER:
        faltan.append("FTP_USER")
    if not FTP_PASS:
        faltan.append("FTP_PASS")
    if not DRIVE_FOLDER_ID:
        faltan.append("DRIVE_FOLDER_ID")
    if not GOOGLE_SERVICE_JSON and not GOOGLE_SERVICE_JSON_PATH:
        faltan.append("GOOGLE_SERVICE_JSON o GOOGLE_SERVICE_JSON_PATH")

    if faltan:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltan)}")

    if not SHP_PATH.exists():
        raise FileNotFoundError(f"No existe el shapefile: {SHP_PATH}")


# ============================================================
# SHAPEFILE
# ============================================================

def read_comids_from_shp(shp_path: Path) -> np.ndarray:
    sf = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in sf.fields[1:]]  # omite DeletionFlag
    field_map = {str(c).strip().lower(): i for i, c in enumerate(fields)}

    if "comid" not in field_map:
        raise ValueError("El shapefile debe tener la columna COMID.")

    idx = field_map["comid"]
    comids = []

    for rec in sf.records():
        try:
            val = rec[idx]
            if val is None or str(val).strip() == "":
                continue
            comids.append(float(val))
        except Exception:
            continue

    if not comids:
        raise ValueError("No se encontraron valores válidos en la columna COMID.")

    return np.array(sorted(set(comids)), dtype="float64")


# ============================================================
# FTP - SOLO LECTURA
# ============================================================

def parse_ftp_modify(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M%S.%f"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass
    return None


def extract_ts_from_name(name: str) -> datetime | None:
    patrones = [
        r"(20\d{12})",  # YYYYMMDDHHMMSS
        r"(20\d{10})",  # YYYYMMDDHHMM
        r"(20\d{8})",   # YYYYMMDDHH
        r"(20\d{6})",   # YYYYMMDD
    ]
    for patron in patrones:
        m = re.search(patron, name)
        if not m:
            continue
        token = m.group(1)
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"):
            try:
                return datetime.strptime(token, fmt)
            except Exception:
                pass
    return None


def list_nc_files(ftp: FTP) -> list[tuple[str, datetime | None]]:
    files: list[tuple[str, datetime | None]] = []

    try:
        for name, facts in ftp.mlsd():
            if name.lower().endswith(".nc"):
                mod = parse_ftp_modify(facts.get("modify"))
                files.append((name, mod))
        if files:
            return files
    except Exception:
        pass

    for name in ftp.nlst():
        if not name.lower().endswith(".nc"):
            continue
        mod = None
        try:
            resp = ftp.sendcmd(f"MDTM {name}")
            token = resp.split()[-1]
            mod = parse_ftp_modify(token)
        except Exception:
            mod = None
        files.append((name, mod))

    return files


def choose_latest_nc(files: list[tuple[str, datetime | None]]) -> str:
    if not files:
        raise FileNotFoundError("No se encontró ningún archivo .nc en el FTP.")

    enriquecidos = []
    for name, mod in files:
        emb = extract_ts_from_name(name)
        score = mod or emb
        enriquecidos.append((name, mod, emb, score))

    con_fecha = [x for x in enriquecidos if x[3] is not None]
    if con_fecha:
        con_fecha.sort(key=lambda x: x[3])
        return con_fecha[-1][0]

    enriquecidos.sort(key=lambda x: x[0].lower())
    return enriquecidos[-1][0]


def read_latest_nc_bytes_from_ftp() -> tuple[bytes, str]:
    with FTP(FTP_HOST) as ftp:
        ftp.login(FTP_USER, FTP_PASS)
        ftp.cwd(FTP_DIR)

        nc_files = list_nc_files(ftp)
        latest_nc = choose_latest_nc(nc_files)

        buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {latest_nc}", buffer.write)
        buffer.seek(0)

        return buffer.getvalue(), latest_nc


# ============================================================
# NETCDF EN MEMORIA
# ============================================================

def orient_tc(arr: np.ndarray, n_t: int, n_c: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"La variable debe ser 2D y llegó con shape {arr.shape}")

    if arr.shape == (n_t, n_c):
        return arr
    if arr.shape == (n_c, n_t):
        return arr.T

    raise ValueError(
        f"Dimensiones inesperadas: {arr.shape}, esperado {(n_t, n_c)} o {(n_c, n_t)}"
    )


def decode_time_var(values: np.ndarray) -> pd.DatetimeIndex:
    vals = np.asarray(values)

    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.to_datetime(vals).normalize()

    if np.issubdtype(vals.dtype, np.number):
        return pd.to_datetime(vals, unit="s", origin="unix").normalize()

    return pd.to_datetime(vals, errors="coerce").normalize()


def open_dataset_from_bytes(nc_bytes: bytes) -> tuple[xr.Dataset, str]:
    errores = []

    for engine in ("h5netcdf", "scipy"):
        try:
            bio = io.BytesIO(nc_bytes)
            ds = xr.open_dataset(bio, engine=engine, chunks=None, cache=False)
            return ds, engine
        except Exception as e:
            errores.append(f"{engine}: {e}")

    raise RuntimeError(
        "No se pudo abrir el NetCDF desde memoria. Errores: " + " | ".join(errores)
    )


def build_filtered_payloads(
    nc_bytes: bytes,
    source_name: str,
    comid_objetivo: np.ndarray,
) -> dict[str, tuple[io.BytesIO, str]]:
    ds, engine_used = open_dataset_from_bytes(nc_bytes)

    try:
        comid_nc = pd.to_numeric(
            pd.Series(np.asarray(ds["comid"].values).ravel()),
            errors="coerce",
        ).to_numpy(dtype="float64")

        time_hist = decode_time_var(ds["time_hist"].values)
        time_frst = decode_time_var(ds["time_frst"].values)

        n_c = len(comid_nc)
        n_t = len(time_hist)
        n_f = len(time_frst)

        idx_keep = np.where(np.isin(comid_nc, comid_objetivo))[0]
        if len(idx_keep) == 0:
            raise ValueError("Ningún COMID del shapefile coincide con el NetCDF.")

        comid_keep = comid_nc[idx_keep]

        qr_hist = orient_tc(np.asarray(ds["qr_hist"].values), n_t, n_c)[:, idx_keep]
        qr_eta_eqm = orient_tc(np.asarray(ds["qr_eta_eqm"].values), n_f, n_c)[:, idx_keep]
        qr_eta_scal = orient_tc(np.asarray(ds["qr_eta_scal"].values), n_f, n_c)[:, idx_keep]
        qr_gfs = orient_tc(np.asarray(ds["qr_gfs"].values), n_f, n_c)[:, idx_keep]
        qr_wrf = orient_tc(np.asarray(ds["qr_wrf"].values), n_f, n_c)[:, idx_keep]

        hist_df = pd.DataFrame({
            "fecha": np.tile(time_hist.to_pydatetime(), len(comid_keep)),
            "comid": np.repeat(comid_keep, len(time_hist)),
            "qr_hist": qr_hist.reshape(-1, order="F"),
        }).sort_values(["comid", "fecha"], kind="stable").reset_index(drop=True)

        fore_df = pd.DataFrame({
            "fecha": np.tile(time_frst.to_pydatetime(), len(comid_keep)),
            "comid": np.repeat(comid_keep, len(time_frst)),
            "qr_eta_eqm": qr_eta_eqm.reshape(-1, order="F"),
            "qr_eta_scal": qr_eta_scal.reshape(-1, order="F"),
            "qr_gfs": qr_gfs.reshape(-1, order="F"),
            "qr_wrf": qr_wrf.reshape(-1, order="F"),
        }).sort_values(["comid", "fecha"], kind="stable").reset_index(drop=True)

        meta = {
            "comid": [float(x) for x in comid_keep.tolist()],
            "ult_fecha_hist": str(pd.to_datetime(time_hist).max().date()),
            "fechas_hist": [str(pd.Timestamp(x).date()) for x in time_hist],
            "fechas_fore": [str(pd.Timestamp(x).date()) for x in pd.to_datetime(time_frst).unique()],
            "nc_origen": f"ftp://{FTP_HOST}{FTP_DIR.rstrip('/')}/{source_name}",
            "nc_nombre": source_name,
            "engine_usado": engine_used,
            "n_comid": int(len(comid_keep)),
        }

        hist_buf = io.BytesIO()
        hist_df.to_parquet(hist_buf, index=False)
        hist_buf.seek(0)

        fore_buf = io.BytesIO()
        fore_df.to_parquet(fore_buf, index=False)
        fore_buf.seek(0)

        meta_buf = io.BytesIO(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
        meta_buf.seek(0)

        return {
            HIST_NAME: (hist_buf, "application/octet-stream"),
            FORE_NAME: (fore_buf, "application/octet-stream"),
            META_NAME: (meta_buf, "application/json"),
        }

    finally:
        ds.close()


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]

    if GOOGLE_SERVICE_JSON:
        info = json.loads(GOOGLE_SERVICE_JSON)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    else:
        with open(GOOGLE_SERVICE_JSON_PATH, "r", encoding="utf-8") as f:
            info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files_in_folder(service, folder_id: str) -> list[dict]:
    q = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None

    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return files


def find_file_in_folder_by_name(service, folder_id: str, filename: str) -> dict | None:
    current_files = list_files_in_folder(service, folder_id)
    for f in current_files:
        if f["name"].lower() == filename.lower():
            return f
    return None


def upload_or_update_buffer(service, folder_id: str, drive_name: str, buffer: io.BytesIO, mime_type: str) -> None:
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype=mime_type, resumable=False)

    existing = find_file_in_folder_by_name(service, folder_id, drive_name)

    if existing:
        service.files().update(
            fileId=existing["id"],
            media_body=media,
            fields="id,name",
            supportsAllDrives=True
        ).execute()
        print(f"Actualizado en Drive: {drive_name}")
    else:
        service.files().create(
            body={"name": drive_name, "parents": [folder_id]},
            media_body=media,
            fields="id,name",
            supportsAllDrives=True
        ).execute()
        print(f"Creado en Drive: {drive_name}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_env()

    print("Leyendo COMID del shapefile...")
    comid_objetivo = read_comids_from_shp(SHP_PATH)
    print(f"COMID válidos: {len(comid_objetivo)}")

    print("Leyendo el .nc más reciente desde FTP a memoria...")
    nc_bytes, source_name = read_latest_nc_bytes_from_ftp()
    print(f"NC leído en memoria: {source_name}")

    print("Procesando NetCDF en memoria...")
    payloads = build_filtered_payloads(nc_bytes, source_name, comid_objetivo)

    print("Conectando a Google Drive...")
    service = get_drive_service()

    print("Actualizando archivos en la misma carpeta de Drive...")
    for drive_name, (buffer, mime_type) in payloads.items():
        upload_or_update_buffer(service, DRIVE_FOLDER_ID, drive_name, buffer, mime_type)

    print("Proceso terminado correctamente.")
    print("Resumen:")
    print(" - FTP solo lectura")
    print(" - NC nunca guardado en disco")
    print(f" - Archivos renovados en Drive: {', '.join(payloads.keys())}")


if __name__ == "__main__":
    main()
