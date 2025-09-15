# File: ies_prep.py (clean, canonical)
from __future__ import annotations

import os, re, json, math, tempfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

__VERSION__ = "2.0.0"

# ===================== Defaults/paths =====================
METADATA_XL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Linear_Data.xlsx")
METADATA_SHEET   = "master_metadata_console"

# ===================== Small utils =====================
def _compact_num(val: Any, places: int = 6) -> str:
    """Compact numeric for IES header (keeps ints tidy, trims trailing zeros)."""
    try:
        x = float(val)
        s = f"{x:.{places}f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except Exception:
        return str(val)

def _fmt_row(nums: List[float], per_line: int = 10) -> List[str]:
    out: List[str] = []
    row: List[str] = []
    for i, x in enumerate(nums, 1):
        row.append(_compact_num(x))
        if i % per_line == 0:
            out.append(" ".join(row)); row = []
    if row:
        out.append(" ".join(row))
    return out

def _safe_name(s: str) -> str:
    s = s or ""
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-","_",".")).strip("._") or "Unknown"

def _normalize_gsheet_id(x: str) -> str:
    """Accept either a bare Sheet ID or a full URL and return the ID."""
    x = (x or "").strip()
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", x)
    return m.group(1) if m else x

# ===================== Step 0: Parse IES =====================
_num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _parse_numbers(s: str) -> List[float]:
    return [float(t) for t in _num_pat.findall(s)]

def parse_ies_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="latin-1") as f:
        txt = f.read()

    # Bracketed metadata [KEY] value
    metadata: Dict[str, Any] = {}
    for m in re.finditer(r"(?m)^\s*(\[[^\]]+\])\s*(.*?)\s*$", txt):
        key = m.group(1).strip(); val = m.group(2).strip()
        if key and key.startswith("[") and key.endswith("]"):
            metadata[key] = val

    # TILT
    m_tilt = re.search(r"(?im)^\s*TILT\s*=\s*([^\r\n]+)", txt)
    if not m_tilt:
        raise ValueError("TILT header missing")
    if m_tilt.group(1).strip().upper() != "NONE":
        raise ValueError("Only TILT=NONE supported")

    # Numeric block
    nums = _parse_numbers(txt[m_tilt.end():])
    it = iter(nums)

    # G0..G12
    header_vals: List[float] = []
    for _ in range(13):
        try:
            header_vals.append(next(it))
        except StopIteration:
            break

    geometry: Dict[str, Any] = {}
    for idx, val in enumerate(header_vals):
        key = f"G{idx}"
        if key in {"G3","G4","G5","G6"}:
            geometry[key] = int(val)
        else:
            geometry[key] = float(val)

    Vn = int(geometry.get("G3", 0)); Hn = int(geometry.get("G4", 0))
    vertical_angles   = [float(next(it)) for _ in range(Vn)]
    horizontal_angles = [float(next(it)) for _ in range(Hn)]

    candela_values: List[List[float]] = []
    for _h in range(Hn):
        candela_values.append([float(next(it)) for _ in range(Vn)])

    data: Dict[str, Any] = {
        "geometry": geometry,
        "metadata": metadata,
        "vertical_angles": vertical_angles,
        "horizontal_angles": horizontal_angles,
        "candela_values": candela_values,  # [H][V]
    }

    # Defaults/derived
    md = data["metadata"]
    md.setdefault("[LIGHT_DIRECTION]", "Direct")
    md["[HEMI_MODE]"] = _detect_hemi_mode(candela_values, vertical_angles)
    data["_flux_original"] = float(calculate_luminous_flux(candela_values, vertical_angles, horizontal_angles))
    return data

def parse_ies_input(path_or_bytes: Any) -> Dict[str, Any]:
    if isinstance(path_or_bytes, str):
        return parse_ies_file(path_or_bytes)
    payload = path_or_bytes.read() if hasattr(path_or_bytes, "read") else path_or_bytes
    txt = payload.decode("latin-1", errors="replace") if isinstance(payload, (bytes, bytearray)) else str(payload)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ies", delete=False, encoding="latin-1") as tmp:
            tmp.write(txt); tmp_path = tmp.name
        return parse_ies_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

# ===================== Photometry math =====================
def calculate_luminous_flux(I_hv, V_deg, H_deg) -> float:
    """
    Integrate luminous flux (Type C) from candela grid.
    Robust for 360° coverage, 180°/90° symmetric sets, and partial H coverage.
    Accepts I as [H][V] or flat [H*V].
    """
    import numpy as _np
    import math as _m

    # ----- Shape inputs -----
    I = _np.asarray(I_hv, dtype=float)
    V = _np.asarray(V_deg, dtype=float)
    H = _np.asarray(H_deg, dtype=float)

    if I.ndim == 1 and V.size * H.size == I.size:
        I = I.reshape(H.size, V.size)
    if I.size == 0 or V.size < 2 or H.size == 0:
        return 0.0

    # ----- Vertical band factors (cosθ_i − cosθ_{i+1}) -----
    Vrad = _np.deg2rad(V)
    dcos = _np.cos(Vrad[:-1]) - _np.cos(Vrad[1:])  # length V-1

    # ----- Horizontal sector widths (dphi) and symmetry factor -----
    if H.size == 1:
        # Single plane => assume full circle
        dphi = _np.array([2.0 * _m.pi], dtype=float)
        I_h  = I  # [1, V]
        sym  = 1.0
    else:
        cover = float(H[-1] - H[0])
        dphi_base = _np.deg2rad(_np.diff(H))  # length H-1

        # Average adjacent H rows so I_h has one row per dphi element
        I_h = 0.5 * (I[:-1, :] + I[1:, :])  # shape [H-1, V]

        # Decide symmetry/wrap
        if abs(cover - 360.0) < 1e-6 or (abs(H[0]) < 1e-6 and abs(H[-1] - 360.0) < 1e-6):
            # Full coverage: append wrap sector and matching averaged row
            wrap = _np.deg2rad((H[0] + 360.0) - H[-1])
            dphi = _np.concatenate([dphi_base, [wrap]])              # length H
            I_wrap = 0.5 * (I[-1, :] + I[0, :])[None, :]             # [1, V]
            I_h = _np.vstack([I_h, I_wrap])                          # [H, V]
            sym = 1.0
        elif abs(H[0]) < 1e-6 and abs(cover - 180.0) < 1e-6:
            # 0..180 coverage → duplicate by symmetry
            dphi = dphi_base                                        # length H-1
            # I_h already length H-1 and aligned with dphi
            sym = 2.0
        elif abs(H[0]) < 1e-6 and abs(cover - 90.0) < 1e-6:
            # 0..90 coverage → fourfold symmetry
            dphi = dphi_base
            sym = 4.0
        else:
            # Arbitrary partial coverage: scale result to full circle
            total = float(dphi_base.sum())
            dphi  = dphi_base
            sym   = (2.0 * _m.pi / total) if total > 0 else 1.0

    # ----- Integrate over H-sectors (rows of I_h) and V-bands -----
    flux = 0.0
    J = I_h.shape[0]            # number of horizontal sectors we have
    for j in range(J):
        phi = float(dphi[j if j < dphi.size else dphi.size - 1])
        for i in range(V.size - 1):
            I_cell = 0.5 * (I_h[j, i] + I_h[j, i + 1])
            flux += I_cell * phi * dcos[i]

    return float(flux) * sym

def _detect_hemi_mode(candela_hv, V_deg) -> str:
    """Return 'D', 'I' or 'B' based on original angle coverage/flux split."""
    V = np.asarray(V_deg, dtype=float)
    if V.size == 0:
        return "D"
    vmin = float(np.nanmin(V)); vmax = float(np.nanmax(V)); eps = 1e-6
    if vmax <= 90.0 + eps:
        return "D"
    if vmin >= 90.0 - eps:
        return "I"
    I = np.asarray(candela_hv, dtype=float)
    if I.ndim == 1 and I.size == V.size:
        I = I.reshape(1, V.size)
    split = int(np.searchsorted(V, 90.0))
    below = float(np.nansum(I[:, :split])); above = float(np.nansum(I[:, split:]))
    tot = below + above
    if tot <= 0: return "D"
    if below / tot >= 0.80: return "D"
    if above / tot >= 0.80: return "I"
    return "B"

# ===================== Step 1: Interpolate + clamp + renorm =====================
def interpolate_candela_matrix(
    data: dict,
    *,
    v_steps: int,
    h_steps: int,
    hemi_mode: str | None = "none",
    clamp_dark: bool = False,
    renorm: bool = False,
) -> dict:
    V_src = np.asarray(data.get("vertical_angles", []) or [], dtype=float)
    H_src = np.asarray(data.get("horizontal_angles", []) or [], dtype=float)
    I_src = np.asarray(data.get("candela_values", []) or [], dtype=float)
    if I_src.ndim == 1 and V_src.size * H_src.size == I_src.size:
        I_src = I_src.reshape(H_src.size, V_src.size)
    if V_src.size == 0 or H_src.size == 0 or I_src.ndim != 2:
        raise ValueError("Bad candela grid")

    V_dst = np.linspace(0.0, 180.0, int(v_steps))
    H_dst = np.linspace(0.0, 360.0, int(h_steps), endpoint=False)

    # Auto mode from original angle coverage
    mode_in = (hemi_mode or "none").strip().lower()
    eps = 1e-6
    if mode_in in {"", "auto"}:
        vmin = float(np.nanmin(V_src)); vmax = float(np.nanmax(V_src))
        if vmax <= 90.0 + eps:
            mode = "direct"; clamp_auto = True
        elif vmin >= 90.0 - eps:
            mode = "indirect"; clamp_auto = True
        else:
            mode = "none"; clamp_auto = False
        clamp = clamp_auto if clamp_dark in (None, False, True) else bool(clamp_dark)
    else:
        mode = mode_in; clamp = bool(clamp_dark)

    # Interpolate V for each H row
    Iv = np.empty((H_src.size, V_dst.size), dtype=float)
    for j in range(H_src.size):
        Iv[j, :] = np.interp(V_dst, V_src, I_src[j, :])

    # Interpolate periodic H for each V column
    Ih = np.empty((H_dst.size, V_dst.size), dtype=float)
    H_pad = np.concatenate([H_src, H_src[:1] + 360.0])
    for i in range(V_dst.size):
        col = Iv[:, i]; col_pad = np.concatenate([col, col[:1]])
        Ih[:, i] = np.interp(H_dst, H_pad, col_pad)

    I_dst = Ih

    # Clamp dark hemisphere if requested
    if clamp:
        if mode == "direct":
            mask = V_dst > 90.0 + eps
        elif mode == "indirect":
            mask = V_dst < 90.0 - eps
        else:
            mask = None
        if mask is not None:
            I_dst[:, mask] = 0.0

    # Flux renorm to match original
    def _flux(dV: np.ndarray, dH: np.ndarray, I_HV: np.ndarray) -> float:
        Vrad = np.deg2rad(dV); cosV = np.cos(Vrad); dcos = cosV[:-1] - cosV[1:]
        Hnext = np.roll(dH, -1); dphi = np.deg2rad((Hnext - dH) % 360.0)
        flux = 0.0
        for j in range(dH.size):
            for i in range(dV.size - 1):
                I_cell = 0.5 * (I_HV[j, i] + I_HV[j, i + 1])
                flux += I_cell * dcos[i] * dphi[j]
        return float(flux)

    flux0 = _flux(V_src, H_src, I_src)
    flux1 = _flux(V_dst, H_dst, I_dst)
    if renorm and flux1 > 0:
        I_dst *= (flux0 / flux1)

    out = dict(data)
    out["vertical_angles"] = V_dst.tolist()
    out["horizontal_angles"] = H_dst.tolist()
    out["candela_values"] = I_dst.astype(float).reshape(-1).tolist()
    # Update hemi metadata
    md = dict(out.get("metadata", {}) or {})
    md["[HEMI_MODE]"] = _detect_hemi_mode(I_dst, V_dst)
    out["metadata"] = md
    return out

# ===================== Step 2: Geometry editor helpers =====================
GEOMETRY_SCHEMA: Dict[str, Dict[str, Any]] = {
    "G5":  {"label": "Photometric type", "editable": False, "tooltip": "1=A, 2=B, 3=C"},
    "G6":  {"label": "Units",            "editable": False, "tooltip": "1=Feet, 2=Meters"},
    "G7":  {"label": "Width (m)",        "editable": True,  "tooltip": "Luminous opening width"},
    "G8":  {"label": "Length (m)",       "editable": True,  "tooltip": "Luminous opening length"},
    "G9":  {"label": "Height (m)",       "editable": True,  "tooltip": "Luminous opening height"},
    "G10": {"label": "System/Input W",   "editable": True,  "tooltip": "System input power"},
    "G11": {"label": "Luminous Flux",    "editable": False, "tooltip": "Computed from photometry"},
    "G12": {"label": "Deprecated",       "editable": False, "tooltip": ""},
    "G13": {"label": "Raw/Lamp lm",      "editable": True,  "tooltip": "Rated lamp lumens"},
    "G14": {"label": "Circuit W",        "editable": True,  "tooltip": "Nameplate/circuit watts"},
}

def _resolve_geom_key(k: str) -> Optional[str]:
    k = (k or "").strip()
    if not k: return None
    if k.upper().startswith("G") and k[1:].isdigit():
        return f"G{int(k[1:])}"
    return None

def apply_geometry_overrides(data: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    if not overrides: return data
    geom = dict(data.get("geometry", {}))
    for raw_k, v in overrides.items():
        gk = _resolve_geom_key(raw_k)
        if not gk: continue
        schema = GEOMETRY_SCHEMA.get(gk, {"editable": True})
        if not schema.get("editable", True):
            continue
        try: geom[gk] = float(v)
        except Exception: geom[gk] = v
    data["geometry"] = geom
    return data

# ===================== Step 2b: Excel mapping =====================

# ---- Google Sheets loader + JSON cache (adds alongside Excel loader) ----
import time

ASSETS_DIR = "assets"
GS_CACHE_JSON = os.path.join(ASSETS_DIR, "linear_data_cache.json")

def _ensure_assets_dir() -> None:
    try:
        os.makedirs(ASSETS_DIR, exist_ok=True)
    except Exception:
        pass

def _schema_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Shared logic to convert a DataFrame into the schema list your UI expects."""
    required = {"FIELD", "IES_ORDER", "IES_FUNC", "IES_TOOLTIP"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] Sheet missing required columns: {missing}")

    df = df.copy()
    df["IES_ORDER"] = pd.to_numeric(df["IES_ORDER"], errors="coerce")
    df = df.dropna(subset=["IES_ORDER"]).sort_values("IES_ORDER")

    schema: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        field = str(r["FIELD"]).strip()
        if not field:
            continue
        func = str(r["IES_FUNC"]).strip()
        tip  = str(r["IES_TOOLTIP"]).strip()
        row: Dict[str, Any] = {
            "order": int(r["IES_ORDER"]),
            "field": field,
            "func": func,
            "tooltip": tip,
        }
        # optional proposed columns
        if "IES_PROPOSED_KEYWORD" in df.columns:
            row["IES_PROPOSED_KEYWORD"] = r.get("IES_PROPOSED_KEYWORD", "")
        if "IES_PROPOSED" in df.columns:
            row["IES_PROPOSED"] = r.get("IES_PROPOSED", "")
        # helpers for UI
        func_l = func.lower()
        row["derived"]  = (func_l == "derived")
        row["editable"] = (func_l == "editable")
        row["geom_key"] = func.upper() if (func.upper().startswith("G") and func[1:].isdigit()) else None
        schema.append(row)
    if not schema:
        raise SystemExit("[FATAL] Schema produced no rows.")
    return schema

def _cache_write_df_json(df: pd.DataFrame, path: str = GS_CACHE_JSON, meta: Optional[Dict[str, Any]] = None) -> None:
    _ensure_assets_dir()
    payload = {
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta": meta or {},
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _cache_read_df_json(path: str = GS_CACHE_JSON) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cols = payload.get("columns") or []
        rows = payload.get("rows") or []
        if cols and rows is not None:
            return pd.DataFrame(rows, columns=cols)
    except Exception:
        return None
    return None

# ===== Google Sheets loader (service account, cached to assets/linear_data_cache.json) =====
GS_CACHE_JSON = os.path.join("assets", "linear_data_cache.json")

def _ensure_assets_dir() -> None:
    try:
        os.makedirs("assets", exist_ok=True)
    except Exception:
        pass

def _schema_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    required = {"FIELD", "IES_ORDER", "IES_FUNC", "IES_TOOLTIP"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] Google Sheet missing required columns: {missing}")
    df = df.copy()
    df["IES_ORDER"] = pd.to_numeric(df["IES_ORDER"], errors="coerce")
    df = df.dropna(subset=["IES_ORDER"]).sort_values("IES_ORDER")
    schema: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        field = str(r["FIELD"]).strip()
        if not field:
            continue
        func = str(r["IES_FUNC"]).strip()
        row: Dict[str, Any] = {
            "order": int(r["IES_ORDER"]),
            "field": field,
            "func": func,
            "tooltip": str(r["IES_TOOLTIP"]).strip(),
            "derived": (func.lower() == "derived"),
            "editable": (func.lower() == "editable"),
            "geom_key": func.upper() if (func.upper().startswith("G") and func[1:].isdigit()) else None,
        }
        if "IES_PROPOSED_KEYWORD" in df.columns:
            row["IES_PROPOSED_KEYWORD"] = r.get("IES_PROPOSED_KEYWORD", "")
        if "IES_PROPOSED" in df.columns:
            row["IES_PROPOSED"] = r.get("IES_PROPOSED", "")
        schema.append(row)
    if not schema:
        raise SystemExit("[FATAL] Schema produced no rows.")
    return schema

def _cache_write_df_json(df: pd.DataFrame, path: str = GS_CACHE_JSON, meta: Optional[Dict[str, Any]] = None) -> None:
    _ensure_assets_dir()
    payload = {
        "fetched_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "meta": meta or {},
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _cache_read_df_json(path: str = GS_CACHE_JSON) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cols = payload.get("columns") or []
        rows = payload.get("rows") or []
        if cols and rows is not None:
            return pd.DataFrame(rows, columns=cols)
    except Exception:
        return None
    return None

def _load_metadata_schema_from_google(sheet_id: str, worksheet_title: str) -> List[Dict[str, Any]]:
    """
    Load schema from a private Google Sheet via service account (gspread).
    Accepts either a bare Sheet ID or the full URL.
    Caches to assets/linear_data_cache.json for offline use.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        raise SystemExit(f"[FATAL] Google Sheets libs missing: {e}. Install: pip install gspread google-auth pandas")

    sid = _normalize_gsheet_id(sheet_id)
    sa_path = os.environ.get("GOOGLE_SA_JSON", "service_account.json")
    if not os.path.exists(sa_path):
        raise SystemExit(f"[FATAL] service_account.json not found at '{sa_path}'. Put your key there or set GOOGLE_SA_JSON.")

    # (Nice error if the worksheet title is wrong)
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    try:
        creds = Credentials.from_service_account_file(sa_path, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sid)
        try:
            ws = sh.worksheet(worksheet_title)
        except gspread.exceptions.WorksheetNotFound:
            titles = [w.title for w in sh.worksheets()]
            raise SystemExit(f"[FATAL] Worksheet '{worksheet_title}' not found. Available tabs: {titles}")

        rows = ws.get_all_records()  # header row used
        if not rows:
            raise SystemExit("[FATAL] Google Sheet returned zero rows.")
        df_live = pd.DataFrame(rows)
        _cache_write_df_json(df_live, GS_CACHE_JSON, meta={"sheet_id": sid, "worksheet": worksheet_title})
        return _schema_from_dataframe(df_live)

    except SystemExit:
        raise
    except Exception as e:
        # Fallback to cache if present
        df_cache = _cache_read_df_json(GS_CACHE_JSON)
        if df_cache is not None:
            return _schema_from_dataframe(df_cache)
        raise SystemExit(f"[FATAL] Google fetch failed and no cache available: {e}")


def _load_metadata_schema_from_excel(xl_path: str, sheet: str) -> List[Dict[str, Any]]:
    if not os.path.exists(xl_path):
        raise SystemExit(f"[FATAL] Metadata Excel not found: {xl_path}")
    try:
        df = pd.read_excel(xl_path, sheet_name=sheet)
    except Exception as e:
        raise SystemExit(f"[FATAL] Failed reading Excel '{xl_path}' sheet '{sheet}': {e}")

    required = {"FIELD", "IES_ORDER", "IES_FUNC", "IES_TOOLTIP"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] Excel sheet missing required columns: {missing}")

    df = df.copy()
    df["IES_ORDER"] = pd.to_numeric(df["IES_ORDER"], errors="coerce")
    df = df.dropna(subset=["IES_ORDER"]).sort_values("IES_ORDER")

    schema: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        field = str(r["FIELD"]).strip()
        if not field: continue
        row: Dict[str, Any] = {
            "order": int(r["IES_ORDER"]),
            "field": field,
            "func": str(r["IES_FUNC"]).strip(),
            "tooltip": str(r["IES_TOOLTIP"]).strip(),
        }
        if "IES_PROPOSED_KEYWORD" in df.columns:
            row["IES_PROPOSED_KEYWORD"] = r.get("IES_PROPOSED_KEYWORD", "")
        if "IES_PROPOSED" in df.columns:
            row["IES_PROPOSED"] = r.get("IES_PROPOSED", "")
        # helpers for UI
        func = row["func"].lower()
        row["derived"]  = (func == "derived")
        row["editable"] = (func == "editable")
        row["geom_key"] = row["func"].upper() if (row["func"].upper().startswith("G") and row["func"][1:].isdigit()) else None
        schema.append(row)
    if not schema:
        raise SystemExit("[FATAL] Excel schema produced no rows after filtering by IES_ORDER.")
    return schema

def apply_metadata_from_schema(
    data: Dict[str, Any],
    schema: List[Dict[str, Any]],
    user_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    geom = data.get("geometry", {}) or {}
    md_old = data.get("metadata", {}) or {}
    md_new: Dict[str, Any] = {}
    locked_fields: set[str] = set()

    for item in schema:
        field = item.get("field", "") or ""
        if not field: continue
        gk = item.get("geom_key")
        if gk:
            md_new[field] = geom.get(gk, "")
            locked_fields.add(field)
            continue
        if field in md_old:
            md_new[field] = md_old[field]
            continue
        proposed = item.get("IES_PROPOSED", "")
        if proposed in ("", None):
            proposed = item.get("IES_PROPOSED_KEYWORD", "")
        md_new[field] = proposed if proposed is not None else ""

    if user_overrides:
        for k, v in user_overrides.items():
            if k in locked_fields: continue
            sch = next((s for s in schema if (s.get("field") or "") == k), None)
            if sch and not bool(sch.get("derived", False)):
                md_new[k] = v

    out = dict(data); out["metadata"] = md_new
    return out

def zonal_flux_split(data: Dict[str, Any]) -> Tuple[float, float]:
    V = list(map(float, data.get("vertical_angles") or []))
    H = list(map(float, data.get("horizontal_angles") or []))
    I = np.array(data.get("candela_values") or [], dtype=float)
    if len(V) < 2 or len(H) < 1 or I.size == 0:
        return (0.0, 0.0)
    if I.ndim == 1 and I.size == len(V) * len(H):
        I = I.reshape(len(H), len(V))
    v_rad = np.radians(np.array(V))
    two_pi_over_H = 2.0 * math.pi / float(len(H))
    f_lo = 0.0; f_hi = 0.0
    for i in range(len(V)-1):
        zone_area = two_pi_over_H * (math.cos(v_rad[i]) - math.cos(v_rad[i+1]))
        seg = 0.5 * (I[:, i] + I[:, i+1]) * zone_area
        v1, v2 = V[i], V[i+1]
        if max(v1, v2) <= 90.0:
            f_lo += float(seg.sum())
        elif min(v1, v2) >= 90.0:
            f_hi += float(seg.sum())
        else:
            frac = (90.0 - min(v1, v2)) / (max(v1, v2) - min(v1, v2))
            f_lo += float(seg.sum()) * frac
            f_hi += float(seg.sum()) * (1.0 - frac)
    return (f_lo, f_hi)

def inject_calculated_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    geom = dict(out.get("geometry", {}) or {})
    meta = dict(out.get("metadata", {}) or {})

    V = np.asarray(out.get("vertical_angles", []) or [], dtype=float)
    H = np.asarray(out.get("horizontal_angles", []) or [], dtype=float)
    I = np.asarray(out.get("candela_values", []) or [], dtype=float)
    if I.ndim == 1 and I.size == V.size * H.size:
        I = I.reshape(H.size, V.size)

    try:
        flux = float(calculate_luminous_flux(I, V, H))
        if not np.isfinite(flux): flux = 0.0
    except Exception:
        flux = 0.0
    geom["G11"] = flux

    def _f(x, default=0.0) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    length_m  = _f(geom.get("G8", 0.0))
    width_m   = _f(geom.get("G7", 0.0))
    height_m  = _f(geom.get("G9", 0.0))
    circuit_w = _f(geom.get("G14", 0.0))
    raw_lm    = _f(geom.get("G13", 0.0))
    ptype     = int(_f(geom.get("G5", 3), 3))

    if circuit_w > 0:
        meta["[LM_PER_W]"] = flux / circuit_w
    if length_m > 0:
        meta["[LM_PER_M]"] = flux / length_m
    if raw_lm > 0:
        meta["[OPT_EFF_PCT]"] = (flux / raw_lm) * 100.0

    meta["[TYPE_LABEL]"] = {1: "Type A", 2: "Type B", 3: "Type C"}.get(ptype, "Type C")
    meta["[LENGTH_M]"]       = length_m
    meta["[WIDTH_M]"]        = width_m
    meta["[HEIGHT_M]"]       = height_m
    meta["[CIRCUIT_WATTS]"]  = circuit_w
    meta["[RAW_LUMENS]"]     = raw_lm

    try:
        down_lm, up_lm = zonal_flux_split(out)
        tol = 1e-6
        if up_lm > tol and down_lm > tol:
            meta["[HEMI_MODE]"] = "B"
        elif up_lm > tol:
            meta["[HEMI_MODE]"] = "I"
        else:
            meta["[HEMI_MODE]"] = "D"
    except Exception:
        meta.setdefault("[HEMI_MODE]", "B")

    out["geometry"] = geom
    out["metadata"] = meta
    return out

# ===================== Step 3: Exchange Hemispheres =====================
def exchange_hemispheres(data: dict) -> dict:
    I = np.asarray(data.get("candela_values", []), dtype=float)
    if I.ndim == 1:
        V = len(data.get("vertical_angles", []) or [])
        H = len(data.get("horizontal_angles", []) or [])
        I = I.reshape(H, V)
    data = deepcopy(data)
    data["candela_values"] = I[:, ::-1].tolist()
    md = dict(data.get("metadata", {}))
    m = str(md.get("[HEMI_MODE]", "D")).upper()
    md["[HEMI_MODE]"] = {"D": "I", "I": "D"}.get(m, "B" if m == "B" else "D")
    data["metadata"] = md
    return data

# ===================== Step 4: Export scaling (1 mm) =====================
def _assert_one_mm_json(data: Dict[str, Any]) -> None:
    try:
        L = float((data.get("geometry", {}) or {}).get("G8", 0.0) or 0.0)
    except Exception:
        L = 0.0
    if abs(L - 0.001) > 1e-9:
        raise ValueError(f"Expected 1 mm JSON (geometry.G8 == 0.001). Got {L!r}.")

def make_scaled_copy_for_export(data: Dict[str, Any], target_len: float = 0.001) -> Dict[str, Any]:
    L = float((data.get("geometry", {}) or {}).get("G8", 1.0) or 1.0)
    s = target_len / L if L else 1.0
    out = deepcopy(data)
    for k in list(out.keys()):
        if isinstance(k, str) and k.startswith("_"):
            out.pop(k, None)

    out["candela_values"] = (np.array(out["candela_values"], dtype=float) * s).tolist()
    geom = dict(out.get("geometry", {}))
    for k in ("G11", "G12", "G13", "G14"):
        geom[k] = float(geom.get(k, 0.0)) * s
    geom["G8"] = target_len
    out["geometry"] = geom
    md = dict(out.get("metadata", {})); md["[LENGTH_M]"] = _compact_num(target_len)
    out["metadata"] = md
    return out

# ===================== Step 5: Merge 1 mm JSONs =====================
def _angles_equal(aV: np.ndarray, aH: np.ndarray, bV: np.ndarray, bH: np.ndarray) -> bool:
    return aV.shape == bV.shape and aH.shape == bH.shape and np.allclose(aV, bV, atol=0.0, rtol=0.0) and np.allclose(aH, bH, atol=0.0, rtol=0.0)

def merge_one_mm_json(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if a is None or b is None:
        raise ValueError("merge_one_mm_json: both inputs required.")
    _assert_one_mm_json(a); _assert_one_mm_json(b)

    V_a = np.array(a.get("vertical_angles", []), dtype=float)
    H_a = np.array(a.get("horizontal_angles", []), dtype=float)
    V_b = np.array(b.get("vertical_angles", []), dtype=float)
    H_b = np.array(b.get("horizontal_angles", []), dtype=float)
    if not _angles_equal(V_a, H_a, V_b, H_b):
        raise ValueError("merge_one_mm_json: angle grids differ; both JSONs must share identical V/H angles.")

    I_a = np.array(a.get("candela_values", []), dtype=float)
    I_b = np.array(b.get("candela_values", []), dtype=float)
    if I_a.ndim == 1 and I_a.size == V_a.size * H_a.size: I_a = I_a.reshape(H_a.size, V_a.size)
    if I_b.ndim == 1 and I_b.size == V_b.size * H_b.size: I_b = I_b.reshape(H_b.size, V_b.size)
    if I_a.shape != I_b.shape:
        raise ValueError(f"merge_one_mm_json: candela shapes differ: {I_a.shape} vs {I_b.shape}")

    I_sum = I_a + I_b
    out: Dict[str, Any] = deepcopy(a)
    out["candela_values"] = I_sum.tolist()
    out["vertical_angles"] = V_a.tolist()
    out["horizontal_angles"] = H_a.tolist()

    geom_a = dict(a.get("geometry", {})); geom_b = dict(b.get("geometry", {}))
    out_geom = dict(geom_a)
    for k in ("G12", "G13", "G14"):
        out_geom[k] = float(geom_a.get(k, 0.0) or 0.0) + float(geom_b.get(k, 0.0) or 0.0)
    out_geom["G3"] = int(len(V_a)); out_geom["G4"] = int(len(H_a)); out_geom["G8"] = 0.001
    out_geom["G11"] = float(calculate_luminous_flux(I_sum, V_a, H_a))
    out["geometry"] = out_geom

    md_a = dict(a.get("metadata", {})); md_b = dict(b.get("metadata", {}))
    out_md = dict(md_a)
    name_a = md_a.get("[LUMCAT]") or md_a.get("[LUMINAIRE]") or md_a.get("[TEST]") or "A"
    name_b = md_b.get("[LUMCAT]") or md_b.get("[LUMINAIRE]") or md_b.get("[TEST]") or "B"
    out_md["[MERGED_FROM]"] = f"{name_a}|{name_b}"
    hemi_a = (md_a.get("[HEMI_MODE]") or "B")[:1].upper(); hemi_b = (md_b.get("[HEMI_MODE]") or "B")[:1].upper()
    out_md["[HEMI_MODE]"] = hemi_a if hemi_a == hemi_b else "B"
    out["metadata"] = out_md
    return out

def scale_one_mm_json(data: Dict[str, Any], target_len_m: float) -> Dict[str, Any]:
    _assert_one_mm_json(data)
    if target_len_m <= 0:
        raise ValueError("scale_one_mm_json: target_len_m must be > 0")
    return make_scaled_copy_for_export(deepcopy(data), target_len=target_len_m)

# ===================== LM-63 File Generation (G10) =====================
class FileGenError(ValueError): pass

@dataclass(frozen=True)
class FileGenFlags:
    accredited: bool = False
    interpolated: bool = False
    scaled: bool = False
    simulated: bool = False
    undefined: bool = False

def compute_file_generation_type(flags: FileGenFlags) -> str:
    if flags.undefined:
        if flags.accredited or flags.interpolated or flags.scaled or flags.simulated:
            raise FileGenError("'Undefined' cannot be combined with other flags.")
        return "1.00001"
    if flags.simulated:
        if flags.accredited or flags.interpolated or flags.scaled or flags.undefined:
            raise FileGenError("'Simulated' cannot be combined with other flags.")
        return "1.00010"
    a = "1" if flags.accredited else "0"
    i = "1" if flags.interpolated else "0"
    s = "1" if flags.scaled else "0"
    return f"1.{a}{i}{s}00"

def file_generation_title(flags: FileGenFlags) -> str:
    if flags.undefined:  return "File generation: Undefined (per LM-63-2019)"
    if flags.simulated:  return "File generation: Simulated (not measured)"
    parts = ["Accredited" if flags.accredited else "Unaccredited"]
    if flags.interpolated: parts.append("Interpolated")
    if flags.scaled:       parts.append("Lumen-scaled")
    return "File generation: " + ", ".join(parts)

def inject_file_generation_type(data: Dict[str, Any], flags: FileGenFlags) -> Dict[str, Any]:
    code = compute_file_generation_type(flags)
    note = file_generation_title(flags)
    out = dict(data)
    geom = dict(out.get("geometry", {})); geom["G10"] = code; out["geometry"] = geom
    md = dict(out.get("metadata", {})); md["[FILEGENINFO]"] = note; out["metadata"] = md
    return out

# ===================== Build IES text =====================
def _filegen_value(geom: Dict[str, Any]) -> str:
    g10 = geom.get("G10")
    s = str(g10).strip() if not isinstance(g10, (int, float)) else str(g10)
    if s.startswith("1"):
        return s
    return "1"

def build_ies_text(data: Dict[str, Any]) -> str:
    md   = dict(data.get("metadata", {}))
    geom = dict(data.get("geometry", {}))

    V = list(map(float, data.get("vertical_angles", []) or []))
    H = list(map(float, data.get("horizontal_angles", []) or []))
    I = np.array(data.get("candela_values", []), dtype=float)

    if I.size == 0 or not V or not H:
        raise ValueError("build_ies_text: missing angles or candela values.")

    lines: List[str] = []
    lines.append("IESNA:LM-63-2002")

    required_order = ["[TEST]", "[TESTLAB]", "[ISSUEDATE]", "[MANUFAC]"]
    for k in required_order:
        if k in md: lines.append(f"{k}={md[k]}")
    for k, v in md.items():
        if k in required_order: continue
        if isinstance(k, str) and k.startswith("[") and k.endswith("]"):
            lines.append(f"{k}={v}")

    lines.append("TILT=NONE")

    n_lamps            = int(geom.get("G0", 1) or 1)
    lumens_per_lamp    = -1
    candela_multiplier = float(geom.get("G2", 1.0) or 1.0)
    nV                 = int(geom.get("G3", len(V)) or len(V))
    nH                 = int(geom.get("G4", len(H)) or len(H))
    p_type             = int(geom.get("G5", 1) or 1)
    units              = int(geom.get("G6", 1) or 1)
    width              = float(geom.get("G7", 0.0) or 0.0)
    length             = float(geom.get("G8", 0.0) or 0.0)
    height             = float(geom.get("G9", 0.0) or 0.0)

    lines.append(" ".join([
        str(n_lamps),
        str(lumens_per_lamp),
        _compact_num(candela_multiplier),
        str(nV),
        str(nH),
        str(p_type),
        str(units),
        _compact_num(width),
        _compact_num(length),
        _compact_num(height),
    ]))

    ballast_factor    = 1.0
    filegen_or_future = _filegen_value(geom)
    input_watts       = float(geom.get("G12", 0.0) or 0.0)
    lines.append(" ".join([_compact_num(ballast_factor), str(filegen_or_future), _compact_num(input_watts)]))

    lines.extend(_fmt_row(V))
    lines.extend(_fmt_row(H))

    if I.ndim == 1 and I.size == len(H) * len(V):
        I = I.reshape(len(H), len(V))
    if I.shape != (len(H), len(V)):
        if I.shape == (len(V), len(H)):
            I = I.T
        else:
            raise ValueError(f"build_ies_text: unexpected candela shape {I.shape}; expected [{len(H)}, {len(V)}].")
    for h in range(len(H)):
        lines.extend(_fmt_row(I[h, :].tolist()))

    return "\n".join(lines) + "\n"

# ===================== Filename helper =====================
def build_filename_from_metadata(metadata: Dict[str, Any], geometry: Dict[str, Any] | None = None) -> str:
    length_m = _compact_num(metadata.get("[LENGTH_M]", (geometry or {}).get("G8", "")), 4)
    lpm     = _compact_num(metadata.get("[LUMENS_PER_M]", ""), 4)
    wpm     = _compact_num(metadata.get("[WATTS_PER_M]", ""), 4)
    lumcat  = _safe_name((metadata.get("[LUMCAT]") or metadata.get("LUMCAT") or "Unknown").strip())
    return f"{length_m}m_{lpm}lm_{wpm}w_{lumcat}"

# ===================== Public API =====================
__all__ = [
    "parse_ies_input",
    "interpolate_candela_matrix",
    "apply_geometry_overrides",
    "_load_metadata_schema_from_excel",
    "apply_metadata_from_schema",
    "inject_calculated_metadata",
    "exchange_hemispheres",
    "make_scaled_copy_for_export",
    "calculate_luminous_flux",
    "build_filename_from_metadata",
    "GEOMETRY_SCHEMA",
    "merge_one_mm_json",
    "scale_one_mm_json",
    "build_ies_text",
    "FileGenFlags",
    "compute_file_generation_type",
    "file_generation_title",
    "inject_file_generation_type",
]


def _load_metadata_schema_from_json_bytes(buf: bytes) -> List[Dict[str, Any]]:
    """
    Accepts the NOVON flat DB JSON as bytes and returns the UI schema
    (same shape as _load_metadata_schema_from_excel/_google).
    Chooses a table with required columns; prefers "master_metadata_console".
    """
    try:
        j = json.loads(buf.decode("utf-8", errors="replace"))
    except Exception as e:
        raise SystemExit(f"[FATAL] Invalid JSON: {e}")

    tables = (j.get("tables") or {})
    if not isinstance(tables, dict) or not tables:
        raise SystemExit("[FATAL] JSON has no 'tables' dict.")

    # Prefer a well-known tab name
    candidates = []
    if "master_metadata_console" in tables:
        candidates = ["master_metadata_console"]
    else:
        candidates = list(tables.keys())

    required = {"FIELD", "IES_ORDER", "IES_FUNC", "IES_TOOLTIP"}
    for name in candidates:
        rows = tables.get(name) or []
        try:
            df = pd.DataFrame(rows)
        except Exception:
            df = pd.DataFrame()
        if not df.empty and required.issubset(set(df.columns)):
            return _schema_from_dataframe(df)

    # As a last resort, try any table that has at least FIELD + IES_ORDER
    relaxed = {"FIELD", "IES_ORDER"}
    for name, rows in tables.items():
        try:
            df = pd.DataFrame(rows)
        except Exception:
            df = pd.DataFrame()
        if not df.empty and relaxed.issubset(set(df.columns)):
            return _schema_from_dataframe(df)

    raise SystemExit("[FATAL] No table with required columns found in JSON.")

