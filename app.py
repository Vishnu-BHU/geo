import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="🧩 GeoRockSlope", page_icon="🪨", layout="centered")

# ----------------------------
# Paths and setup
# ----------------------------
BASE = Path(__file__).parent.resolve()
MODELS_DIR = BASE / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"
RANGES_PATH = BASE / "training_ranges.json"

# ----------------------------
# Logos
# ----------------------------
LOGO_URL_1 = "https://github.com/Vishnu-BHU/geo/blob/main/assets/ANRF.png?raw=true"
LOGO_URL_2 = "https://github.com/Vishnu-BHU/geo/blob/main/assets/GECL.png?raw=true"
LOGO_URL_3 = "https://github.com/Vishnu-BHU/geo/blob/main/assets/bhu.png?raw=true"

# ----------------------------
# Constants
# ----------------------------
FEATURE_ORDER = ["SlopeHeight", "SlopeAngle", "UCS", "GSI", "mi", "D", "PoissonsRatio", "E", "Density"]

INPUT_LABELS = {
    "MODEL": "Prediction Model",
    "SLOPE_HEIGHT": "Slope Height",
    "SLOPE_ANGLE": "Slope Angle",
    "UCS": "Uniaxial Compressive Strength",
    "GSI": "Geological Strength Index",
    "MI": "Material Constant (mi)",
    "D_VAL": "Disturbance Factor",
    "PR": "Poisson's Ratio",
    "YM": "Young’s Modulus (E) of Intact Rock",
    "DEN": "Density",
}

DEFAULT_BOUNDS = {
    "SlopeHeight": (13.0, 74.0),
    "SlopeAngle": (55.0, 84.0),
    "UCS": (42.0, 87.0),
    "GSI": (25, 85),
    "mi": (23, 35),
    "PoissonsRatio": (0.15, 0.22),
    "E": (8783.0, 36123.0),
    "Density": (2.55, 2.75),
}

UNITS = {
    "SlopeHeight": " m",
    "SlopeAngle": "°",
    "UCS": " MPa",
    "GSI": "",
    "mi": "",
    "D": "",
    "PoissonsRatio": "",
    "E": " MPa",
    "Density": " g/cm³",
}

D_VALS = {
    "Moderately Disturbed Rock Mass": 0.7,
    "Very Disturbed Rock Mass": 1.0,
}

SAT_FACTOR_LOW = 0.821
SAT_FACTOR_HIGH = 0.881

# ----------------------------
# Header with multiple logos
# ----------------------------
def header_with_logo(title: str = "GeoRockSlope", logo_width: int = 96, logo_urls: list[str] | None = None):
    """Display title and one or more logos in a single header row."""
    if logo_urls is None:
        logo_urls = [LOGO_URL_1]

    # Allocate column weights: larger for title, smaller evenly for logos
    title_weight = 0.8
    if len(logo_urls) > 0:
        remaining = 1.0 - title_weight
        logo_weight = remaining / len(logo_urls)
        weights = [title_weight] + [logo_weight] * len(logo_urls)
    else:
        weights = [1.0]

    cols = st.columns(weights)
    with cols[0]:
        st.markdown(f"<h1 style='margin:0'>{title}</h1>", unsafe_allow_html=True)
    for i, url in enumerate(logo_urls):
        with cols[i + 1]:
            st.image(url, width=logo_width)

# ----------------------------
# Persistent RNG for saturated estimate
# ----------------------------
def _get_rng(seed: int | None):
    if "sat_rng" not in st.session_state or (seed is not None and st.session_state.get("sat_seed") != seed):
        st.session_state["sat_seed"] = seed
        st.session_state["sat_rng"] = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    return st.session_state["sat_rng"]

# ----------------------------
# Loaders
# ----------------------------
@st.cache_resource
def load_ranges():
    if RANGES_PATH.exists():
        try:
            return json.loads(RANGES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def _pretty_model_name(folder_name: str) -> str:
    s = folder_name.lower()
    if s.startswith("abc_"):
        algo = "Artificial Bee Colony"
    elif s.startswith("ga_"):
        algo = "Genetic Algorithm"
    elif s.startswith("acor_"):
        algo = "Ant Colony Optimization (ACOR)"
    else:
        algo = folder_name.replace("_", " ").title()
    seismic = " (Seismic)" if ("sf" in s or "seismic" in s) else ""
    return f"{algo}{seismic}"

@st.cache_resource
def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "models" in data:
                return data["models"]
            if isinstance(data, list):
                return data
        except Exception as e:
            st.warning(f"Manifest read error: {e}")

    entries = []
    if MODELS_DIR.exists():
        for sub in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
            m, sx, sy = sub / "model.joblib", sub / "scaler_X.joblib", sub / "scaler_y.joblib"
            if m.exists() and sx.exists() and sy.exists():
                entries.append({
                    "id": sub.name,
                    "name": _pretty_model_name(sub.name),
                    "model_path": str(m),
                    "scaler_X_path": str(sx),
                    "scaler_y_path": str(sy),
                    "target_name": "Seismic FoS" if ("sf" in sub.name.lower() or "seismic" in sub.name.lower()) else "FoS",
                    "feature_names": FEATURE_ORDER,
                })
    return entries

@st.cache_resource(show_spinner=False)
def load_artifacts(entry):
    model = joblib.load(entry["model_path"])
    scaler_X = joblib.load(entry["scaler_X_path"])
    scaler_y = joblib.load(entry["scaler_y_path"])
    return model, scaler_X, scaler_y

# ----------------------------
# Input helpers
# ----------------------------
def get_bounds(name, ranges_data):
    if ranges_data and "ranges" in ranges_data and name in ranges_data["ranges"]:
        r = ranges_data["ranges"][name]
        return float(r["min"]), float(r["max"])
    return DEFAULT_BOUNDS.get(name, (None, None))

def rng_help(name, ranges_data):
    mn, mx = get_bounds(name, ranges_data)
    if mn is None or mx is None:
        return ""
    unit = UNITS.get(name, "")
    return f"Training range: {mn:g} to {mx:g}{unit}"

def int_input(label, mn, mx, val, help_txt):
    return st.number_input(label, min_value=int(mn), max_value=int(mx), value=int(val), step=1, format="%d", help=help_txt)

def float_input(label, mn, mx, val, step, fmt, help_txt, epsilon=0.0):
    return st.number_input(
        label=label,
        min_value=float(mn),
        max_value=float(mx) + float(epsilon),
        value=float(val),
        step=float(step),
        format=fmt,
        help=help_txt,
    )

def render_inputs(feature_names, ranges_data):
    colLeft, colRight = st.columns(2)
    vals = {}

    mn, mx = get_bounds("SlopeHeight", ranges_data)
    with colLeft:
        vals["SlopeHeight"] = float_input(INPUT_LABELS["SLOPE_HEIGHT"], mn, mx, mn, 0.1, "%.1f", rng_help("SlopeHeight", ranges_data))
    mn, mx = get_bounds("SlopeAngle", ranges_data)
    with colLeft:
        vals["SlopeAngle"] = float_input(INPUT_LABELS["SLOPE_ANGLE"], mn, mx, mn, 0.1, "%.1f", rng_help("SlopeAngle", ranges_data))
    mn, mx = get_bounds("UCS", ranges_data)
    with colLeft:
        vals["UCS"] = float_input(INPUT_LABELS["UCS"], mn, mx, mn, 0.1, "%.1f", rng_help("UCS", ranges_data))
    mn, mx = get_bounds("GSI", ranges_data)
    with colLeft:
        vals["GSI"] = int_input(INPUT_LABELS["GSI"], mn, mx, mn, rng_help("GSI", ranges_data))

    mn, mx = get_bounds("mi", ranges_data)
    with colRight:
        vals["mi"] = int_input(INPUT_LABELS["MI"], mn, mx, mn, rng_help("mi", ranges_data))

    vals["D"] = D_VALS[st.selectbox(INPUT_LABELS["D_VAL"], list(D_VALS.keys()), help=rng_help("D", ranges_data))]

    mn, mx = get_bounds("PoissonsRatio", ranges_data)
    with colRight:
        vals["PoissonsRatio"] = float_input(INPUT_LABELS["PR"], mn, mx, mn, 0.01, "%.2f", rng_help("PoissonsRatio", ranges_data), epsilon=1e-9)

    mn, mx = get_bounds("E", ranges_data)
    with colRight:
        vals["E"] = float_input(INPUT_LABELS["YM"], mn, mx, mn, 0.1, "%.1f", rng_help("E", ranges_data))
    mn, mx = get_bounds("Density", ranges_data)
    with colRight:
        vals["Density"] = float_input(INPUT_LABELS["DEN"], mn, mx, mn, 0.01, "%.2f", rng_help("Density", ranges_data))

    x_row = [float(vals[n]) for n in feature_names]
    return vals, x_row

def predict_one(model, scaler_X, scaler_y, row_vals):
    X = np.array(row_vals, dtype=float).reshape(1, -1)
    Xs = scaler_X.transform(X)
    y_scaled = model.predict(Xs).reshape(-1, 1)
    y = scaler_y.inverse_transform(y_scaled).ravel()
    return float(y[0])

# ----------------------------
# Page layout
# ----------------------------
header_with_logo(logo_urls=[LOGO_URL_1, LOGO_URL_2, LOGO_URL_3], logo_width=96)

ranges = load_ranges()
models = load_manifest()

if not models:
    st.error("No models found. Each folder in 'models/' must contain model.joblib, scaler_X.joblib, scaler_y.joblib.")
    st.caption(f"Looking in: {MODELS_DIR}")
    st.stop()

choices = {m["name"]: m for m in models}
chosen = st.selectbox(INPUT_LABELS["MODEL"], list(choices.keys()))
entry = choices[chosen]

model, scaler_X, scaler_y = load_artifacts(entry)
feature_names = entry.get("feature_names", FEATURE_ORDER)
target_name = entry.get("target_name", "FoS")

is_seismic = "seismic" in target_name.lower()
if is_seismic:
    st.checkbox("Estimate FoS under Saturated condition", value=False, disabled=True, key="sat_disabled_view", help="Unavailable for Seismic models")
    use_saturated_estimate = False
else:
    use_saturated_estimate = st.checkbox("Estimate FoS under Saturated condition", value=False, key="use_saturated_estimate")

values, x_row = render_inputs(feature_names, ranges)

if hasattr(scaler_X, "n_features_in_") and scaler_X.n_features_in_ != len(x_row):
    st.error(f"Feature count mismatch: scaler expects {scaler_X.n_features_in_}, got {len(x_row)}")
else:
    if st.button(f"Predict {target_name}", type="primary"):
        try:
            y = predict_one(model, scaler_X, scaler_y, x_row)
            if use_saturated_estimate and not is_seismic:
                rng = _get_rng(None)
                low, high = y * SAT_FACTOR_LOW, y * SAT_FACTOR_HIGH
                y_sat = float(rng.uniform(low, high))
                st.success(f"Saturated FoS: **{y_sat:.4f}**")
            else:
                st.success(f"Predicted {target_name}: **{y:.4f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
