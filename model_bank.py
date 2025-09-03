# model_bank.py
# ---------------------------------------------------------
# Route predictions to the best-trained bank model for *any*
# user-provided subset (>=3) by aliasing user column names
# to canonical training features. Falls back to LGBM superset
# (if saved) when no bank model matches.
# ---------------------------------------------------------

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
# --- at the top of model_bank.py (after other imports)
# ---- put near the top (after other imports) ----
import os, pickle, requests, joblib

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
REGISTRY_PATH = os.path.join(APP_DIR, "model_registry.json")
os.makedirs(MODEL_DIR, exist_ok=True)

def _get_secrets():
    """Fetch MODELS_BASE_URL/HF_TOKEN at call-time (works on Streamlit Cloud)."""
    base  = os.environ.get("MODELS_BASE_URL") or ""
    token = os.environ.get("HF_TOKEN") or ""
    try:
        import streamlit as st
        base  = base  or st.secrets.get("MODELS_BASE_URL", "")
        token = token or st.secrets.get("HF_TOKEN", "")
    except Exception:
        pass
    return base.strip(), token.strip()

def _remote_urls(model_name: str, base_url: str):
    base = base_url.rstrip("/")
    return [f"{base}/{model_name}.pkl", f"{base}/{model_name}.joblib"]

def _download(url: str, out_path: str, token: str = ""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)

def load_model(model_name: str):
    """Load from ./models; if missing, download from MODELS_BASE_URL and cache."""
    local_pkl = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    local_jl  = os.path.join(MODEL_DIR, f"{model_name}.joblib")

    # 1) local cache first
    if os.path.exists(local_pkl):
        with open(local_pkl, "rb") as f:
            return pickle.load(f)
    if os.path.exists(local_jl):
        return joblib.load(local_jl)

    # 2) remote fetch (call-time secrets)
    base_url, token = _get_secrets()
    if base_url:
        last_err = None
        for url in _remote_urls(model_name, base_url):
            try:
                out = local_pkl if url.endswith(".pkl") else local_jl
                _download(url, out, token)
                return pickle.load(open(out, "rb")) if out.endswith(".pkl") else joblib.load(out)
            except Exception as e:
                last_err = e
                continue
        raise FileNotFoundError(
            f"Tried to download {model_name} (.pkl/.joblib) but failed. "
            f"Check MODELS_BASE_URL and that the file exists remotely. Last error: {last_err}"
        )

    # 3) no base URL available
    raise FileNotFoundError(f"Trained model file not found locally and no MODELS_BASE_URL set: {model_name}")

# --- Registry helpers (add these) ---
def _read_registry_lines(path: str):
    """Read newline-delimited JSON (one model per line)."""
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    # normalize feature names to UPPER for matching
                    feats = [str(x).upper() for x in obj.get("features", [])]
                    obj["features"] = feats
                    items.append(obj)
            except Exception:
                continue
    return items

def load_registry() -> dict:
    """
    Returns {"models": [...]} from model_registry.json.
    Supports:
      - newline-delimited JSON (one object per line)
      - a JSON list
      - a JSON dict with "models"
    """
    path = REGISTRY_PATH
    if not os.path.exists(path):
        return {"models": []}

    # Try full-file JSON first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "models" in data:
            # normalize features to UPPER
            for e in data["models"]:
                if isinstance(e, dict):
                    e["features"] = [str(x).upper() for x in e.get("features", [])]
            return data
        if isinstance(data, list):
            for e in data:
                if isinstance(e, dict):
                    e["features"] = [str(x).upper() for x in e.get("features", [])]
            return {"models": data}
    except Exception:
        pass

    # Fallback: line-by-line registry
    return {"models": _read_registry_lines(path)}

# -----------------------------
# Canonical feature families
# -----------------------------
# IMPORTANT: These are the canonical names your bank expects.
# We map user column names to these.
CANON_FAMILIES: Dict[str, List[str]] = {
    # Gamma
    "GR":   [r"\bgr\b", r"gamma[_\s-]*ray", r"\bgamma\b"],
    "SGR":  [r"\bsgr\b", r"spectral[_\s-]*gamma"],

    # Density
    "RHOB": [r"\brhob\b", r"\brho\b", r"\bdensity\b", r"\bden\b", r"bulk[_\s-]*density"],
    "RHOM": [r"\brhom\b", r"matrix[_\s-]*density", r"\brho[_\s-]*ma\b"],
    "DRHO": [r"\bdrho\b", r"delta[_\s-]*rho", r"density[_\s-]*correction"],

    # Neutron
    "NPHI": [r"\bnphi\b", r"neutron[_\s-]*porosity", r"\bneu\b", r"\bneutron\b"],

    # Sonic (treat as interchangeable family)
    "DT":    [r"\bdt\b", r"\bdtc\b", r"\bac\b", r"\bsonic\b", r"slowness"],
    "DTC":   [r"\bdtc\b", r"\bdt\b", r"\bac\b", r"\bsonic\b"],
    "AC":    [r"\bac\b", r"\bdt\b", r"\bdtc\b", r"\bsonic\b"],
    "SONIC": [r"\bsonic\b", r"\bdt\b", r"\bdtc\b", r"\bac\b"],

    # PEF
    "PEF":  [r"\bpef\b", r"\bpe\b", r"photoelectric"],

    # Resistivity family (interchangeable within family)
    "RT":    [r"resist", r"\bohm", r"\bdeep\b", r"\brt\b", r"\bild\b", r"\blld\b"],
    "RDEP":  [r"resist", r"\bdeep\b", r"\brdep\b", r"\brt\b", r"\bild\b", r"\blld\b"],
    "RMED":  [r"resist", r"\bmed(i?um)?\b", r"\brmed\b", r"\blls\b"],
    "RLLD":  [r"resist", r"\bdeep\b", r"\brlld\b", r"\blld\b"],
    "RLLS":  [r"resist", r"\bshallow\b", r"\brlls\b", r"\bmsfl\b", r"\bsfl\b"],
    "ILD":   [r"\bild\b", r"deep[_\s-]*res"],
    "LLD":   [r"\blld\b", r"deep[_\s-]*res"],
    "LLS":   [r"\blls\b", r"shallow[_\s-]*res"],
    "MSFL":  [r"\bmsfl\b", r"micro(?:-?sfl)?", r"shallow[_\s-]*res"],
}

# Equivalence groups (a required name can be satisfied by *any* in its group)
SONIC_EQ: Set[str] = {"DT", "DTC", "AC", "SONIC"}
RES_EQ:   Set[str] = {"RT","RDEP","RMED","RLLD","RLLS","ILD","LLD","LLS","MSFL"}
# You can add more groups if you like; we keep RHOM separate on purpose.

def _normalize_name(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r"[^\w]+", "_", s)

def _candidate_canon_for_user_col(colname: str) -> Set[str]:
    """Return all canonical names this user column could represent."""
    norm = _normalize_name(colname)
    hits = set()
    for canon, patterns in CANON_FAMILIES.items():
        for pat in patterns:
            if re.search(pat, norm):
                hits.add(canon)
                break
    return hits

def make_alias_map(user_df: pd.DataFrame) -> Tuple[Dict[str, str], Set[str], List[str]]:
    """
    From user columns -> canonical aliases.
    Returns:
      - canon_to_user: {CANON -> user_col_name}
      - available_canon: set of canonical names we can synthesize
      - mapping_info: list strings for UI ("user_col -> [canon...]")
    One user column may satisfy multiple canonical names; that's OK.
    """
    canon_to_user: Dict[str, str] = {}
    mapping_info: List[str] = []

    for col in user_df.columns:
        cands = _candidate_canon_for_user_col(col)
        if not cands:
            continue
        # Set mappings for all *unclaimed* canon names this column can satisfy.
        for canon in cands:
            if canon not in canon_to_user:
                canon_to_user[canon] = col
        mapping_info.append(f"{col} → {sorted(list(cands))}")

    available_canon = set(canon_to_user.keys())
    return canon_to_user, available_canon, mapping_info

# -----------------------------
# Model selection with aliases
# -----------------------------
def _required_is_covered(required: List[str], available_canon: Set[str]) -> bool:
    """
    A model is compatible if every required name is either directly in available_canon
    or is satisfied by an equivalent family member (sonic / resistivity).
    """
    for req in (str(r).upper() for r in required):
        if req in available_canon:
            continue
        # Family equivalences
        if req in SONIC_EQ and SONIC_EQ.intersection(available_canon):
            continue
        if req in RES_EQ and RES_EQ.intersection(available_canon):
            continue
        # Not covered
        return False
    return True

def select_best_entry_for_available(features_available, registry):
    """
    Choose among models whose required features are covered by the user's columns (via aliases).
    Sort priority:
      1) more required features (coverage)
      2) higher OOB score (if present)
      3) higher CV F1 (cv_f1_macro), if present
      4) higher in-sample F1 (f1_macro)
    """
    avail = set(map(str.upper, features_available))
    candidates = []
    for entry in registry.get("models", []):
        req = [str(x).upper() for x in entry.get("features", [])]
        if not _required_is_covered(req, avail):
            continue
        m = entry.get("metrics", {}) or {}
        # Pull metrics if present; NaN means "unknown"
        oob  = m.get("oob_score", float("nan"))
        cvf1 = m.get("cv_f1_macro", float("nan"))
        f1   = m.get("f1_macro", 0.0) or 0.0
        # Store tuple used for sorting
        candidates.append((len(req), oob, cvf1, f1, entry))

    if not candidates:
        return None

    def key(t):
        nfeat, oob, cvf1, f1, _ = t
        # For sorting, missing scores (NaN) should sort *after* real numbers
        oob_key  = (-oob)  if np.isfinite(oob)  else float("+inf")
        cvf1_key = (-cvf1) if np.isfinite(cvf1) else float("+inf")
        return (-nfeat, oob_key, cvf1_key, -f1)

    candidates.sort(key=key)
    return candidates[0][4]

# -----------------------------
# Predict (routes to best match)
# -----------------------------
def _assemble_X_in_required_order(df: pd.DataFrame, required: List[str], canon_to_user: Dict[str, str]) -> np.ndarray:
    """
    Build X in the model's exact required order by pulling from alias map.
    If a required feature is in a family (sonic or resistivity), use any available member
    from that family. Duplicates a user column if it satisfies multiple required names.
    """
    cols = []
    for req in [str(r).upper() for r in required]:
        # direct alias
        src = canon_to_user.get(req)

        # family fallback (sonic)
        if src is None and req in SONIC_EQ:
            for fam in ["DT","DTC","AC","SONIC"]:
                if canon_to_user.get(fam):
                    src = canon_to_user[fam]; break

        # family fallback (resistivity)
        if src is None and req in RES_EQ:
            for fam in ["RT","RDEP","RMED","RLLD","RLLS","ILD","LLD","LLS","MSFL"]:
                if canon_to_user.get(fam):
                    src = canon_to_user[fam]; break

        # final fallback: if the user already has an exact column named like req
        if src is None and req in df.columns:
            src = req

        if src is None:
            raise ValueError(f"Could not map required feature '{req}' to any user column.")

        col_series = pd.to_numeric(df[src], errors="coerce")
        cols.append(col_series)

    X = pd.concat(cols, axis=1).values
    return X

def predict_with_best_model(df: pd.DataFrame, features: List[str] = None):
    """
    Main entry the interface calls. 'features' is optional here; we infer aliases from df columns.
    Returns (predictions, info_dict)
    info_dict contains model_name, required_features, metrics, mapping_info (for UI).
    """
    registry = load_registry()

    # 1) Build alias mapping from the *uploaded* DataFrame column names
    canon_to_user, available_canon, mapping_info = make_alias_map(df)

    # 2) Pick the best entry compatible with our available canonical set
    best_entry = select_best_entry_for_available(available_canon, registry)

    # 3) If bank has no compatible model, optionally fall back to LGBM superset
    if best_entry is None:
        superset_meta = os.path.join(MODEL_DIR, "lgbm_superset_meta.json")
        superset_model = os.path.join(MODEL_DIR, "lgbm_superset.pkl")
        if os.path.exists(superset_meta) and os.path.exists(superset_model):
            with open(superset_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feats = meta.get("feature_list", [])
            with open(superset_model, "rb") as f:
                mdl = pickle.load(f)
            X = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").values
            preds = mdl.predict(X)
            return preds, {
                "model_name": "lgbm_superset.pkl",
                "required_features": feats,
                "metrics": {"f1_macro": float("nan")},
                "mapping_info": mapping_info
            }
        # If no superset available, error with guidance
        raise ValueError(
            "No suitable model found for the provided feature names. "
            "Try renaming your columns to standard log names (e.g., GR, RHOB, NPHI, DT/DTC, PEF, RT/RDEP/LLD/LLS) "
            "or include at least 3 lithology logs."
        )

    # 4) Load the chosen model and assemble X in its required order via aliases
    model_name = best_entry["model_name"]
    required = [str(r).upper() for r in best_entry.get("features", [])]
    model = load_model(model_name)

    X = _assemble_X_in_required_order(df, required, canon_to_user)
    preds = model.predict(X)

    info = dict(best_entry)
    info["required_features"] = required
    info["mapping_info"] = mapping_info
    return preds, info

# -----------------------------
# Optional helper for your UI
# -----------------------------
def list_compatible_models_for_df(df: pd.DataFrame) -> List[Tuple[str, float, int]]:
    """
    Returns a quick list of (model_name, f1_macro, n_required) that match the given dataframe columns.
    Useful for debugging in the interface.
    """
    reg = load_registry()
    canon_to_user, avail, _ = make_alias_map(df)
    out = []
    for e in reg.get("models", []):
        req = e.get("features", [])
        if _required_is_covered(req, avail):
            f1 = float(((e.get("metrics", {}) or {}).get("f1_macro", 0.0)) or 0.0)
            out.append((e["model_name"], f1, len(req)))
    # sort by F1 desc, then fewer required features
    out.sort(key=lambda t: (-t[1], t[2]))
    return out
