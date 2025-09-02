
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Well Log Plotter + Lithology Predictor", layout="wide")
st.title("Well Log Plotter + Lithology Predictor")

LOG_BUTTONS = ["THOR","SGR","NPHI","RHOB","PEF","GR","URAN","DTC","RDEP","SP","RSHA","RMIC","RHOM","RMED"]
DEPTH_CANDIDATES = ["DEPT","DEPTH"]

# -------- Helpers --------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def detect_depth(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def lttb_downsample(x: np.ndarray, y: np.ndarray, n_out: int):
    """Largest-Triangle-Three-Buckets downsampling (preserves shape)."""
    n = len(x)
    if n_out >= n or n_out < 3:
        return x, y
    idx = np.arange(n, dtype=np.int64)

    # Bucket size
    bucket_size = (n - 2) / (n_out - 2)

    # Always include first & last points
    a = 0
    sampled_idx = [0]

    for i in range(1, n_out - 1):
        # range for this bucket
        start = int(np.floor((i - 1) * bucket_size) + 1)
        end = int(np.floor(i * bucket_size) + 1)
        end = min(end, n - 1)

        bucket_x = x[start:end]
        bucket_y = y[start:end]
        if bucket_x.size == 0:
            continue

        # next bucket to choose avg from
        next_start = int(np.floor(i * bucket_size) + 1)
        next_end = int(np.floor((i + 1) * bucket_size) + 1)
        next_end = min(next_end, n)
        avg_x = np.mean(x[next_start:next_end]) if next_start < next_end else x[end]
        avg_y = np.mean(y[next_start:next_end]) if next_start < next_end else y[end]

        ax, ay = x[a], y[a]
        bx = bucket_x
        by = bucket_y
        area = np.abs((ax - avg_x) * (by - ay) - (bx - ax) * (avg_y - ay))
        chosen = np.argmax(area)
        a = start + chosen
        sampled_idx.append(a)

    sampled_idx.append(n - 1)
    sampled_idx = np.array(sorted(set(sampled_idx)), dtype=np.int64)
    return x[sampled_idx], y[sampled_idx]

def downsample_df(df, depth_col, cols, max_points):
    if max_points is None or max_points <= 0:
        return df[[depth_col] + cols].copy()
    out = {depth_col: None}
    x = df[depth_col].to_numpy()
    xs, _ = lttb_downsample(x, x, max_points)  # we only need indices pattern once
    xs_set = set(xs.tolist())
    mask = np.array([v in xs_set for v in x])
    out[depth_col] = df.loc[mask, depth_col].to_numpy()
    for c in cols:
        out[c] = df.loc[mask, c].to_numpy()
    return pd.DataFrame(out)

# -------- State --------
if "selected_cols" not in st.session_state:
    st.session_state.selected_cols = []

def toggle_selection(col_name: str):
    if col_name in st.session_state.selected_cols:
        st.session_state.selected_cols.remove(col_name)
    else:
        st.session_state.selected_cols.append(col_name)

# -------- Sidebar: Data --------
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV with your logs", type=["csv"])
if uploaded is not None:
    df = load_csv(uploaded)
else:
    st.sidebar.info("No file uploaded — using a tiny demo DataFrame.")
    depth = np.arange(1000, 11000, 0.2)  # big demo to show speedup
    rng = depth.size
    df = pd.DataFrame({
        "DEPT": depth,
        "GR": np.clip(np.random.normal(80, 10, size=rng), 0, 200),
        "RHOB": np.random.normal(2.35, 0.05, size=rng),
        "NPHI": np.random.normal(0.3, 0.05, size=rng),
        "RDEP": np.abs(np.random.normal(10, 3, size=rng)),
        "SP": np.random.normal(-50, 5, size=rng),
    })

depth_col = detect_depth(df, DEPTH_CANDIDATES)
if depth_col is None:
    st.warning("No depth-like column found (tried: DEPT, DEPTH). Using row index.")
else:
    st.caption(f"Detected depth column: **{depth_col}**")

# -------- Sidebar: Plot Options --------
st.sidebar.header("2) Plot Options")
renderer = st.sidebar.selectbox(
    "Renderer",
    ["Matplotlib (static, fastest)", "Plotly WebGL (interactive)"],
    index=0
)
invert_depth = st.sidebar.checkbox("Invert depth axis (depth increases downward)", value=True)
normalize = st.sidebar.checkbox("Normalize selected logs (min–max)", value=False)
max_points = st.sidebar.number_input(
    "Max points per curve (downsampling)",
    min_value=0, value=5000, step=1000,
    help="0 = no downsampling. LTTB keeps the visual shape while reducing points."
)

# -------- Buttons grid --------
cols_available = [c for c in LOG_BUTTONS if c in df.columns]
cols_missing = [c for c in LOG_BUTTONS if c not in df.columns]
if cols_missing:
    st.sidebar.caption(f"Unavailable in your data: {', '.join(cols_missing)}")

st.subheader("Select Logs (click to toggle)")
GRID_W = 7
rows = (len(cols_available) + GRID_W - 1) // GRID_W
for r in range(rows):
    row_cols = st.columns(GRID_W)
    for i in range(GRID_W):
        idx = r * GRID_W + i
        if idx >= len(cols_available): break
        name = cols_available[idx]
        active = name in st.session_state.selected_cols
        label = f"✅ {name}" if active else f"➕ {name}"
        if row_cols[i].button(label, key=f"btn_{name}"):
            toggle_selection(name)

left, right = st.columns([1,1])
with left:
    if st.button("Clear selection"):
        st.session_state.selected_cols = []
with right:
    st.write("Selected:", ", ".join(st.session_state.selected_cols) if st.session_state.selected_cols else "—")

if not st.session_state.selected_cols:
    st.info("Pick one or more logs above to draw their graphs.")
else:
    # -------- Prepare data for plotting --------
    if depth_col:
        plot_df = df[[depth_col] + st.session_state.selected_cols].dropna().copy()
        plot_df.sort_values(depth_col, inplace=True)
    else:
        plot_df = df[st.session_state.selected_cols].dropna().copy()
        plot_df.reset_index(inplace=True)
        depth_col = "index"  # synthetic

    # Normalize
    if normalize:
        for c in st.session_state.selected_cols:
            col_min = plot_df[c].min()
            col_max = plot_df[c].max()
            if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
                plot_df[c] = (plot_df[c] - col_min) / (col_max - col_min)
    # ensure numeric for depth + selected logs (like the training script)
    cols_to_numeric = [depth_col] + list(st.session_state.selected_cols)
    for c in cols_to_numeric:
        if c in plot_df.columns:
            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Downsample
    if max_points and max_points > 0:
        plot_df = downsample_df(plot_df, depth_col, st.session_state.selected_cols, int(max_points))

    st.subheader("Graphs")

    # -------- Renderers --------
    if renderer.startswith("Matplotlib"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
        for c in st.session_state.selected_cols:
            ax.plot(plot_df[c].to_numpy(), plot_df[depth_col].to_numpy(), linewidth=1)
        ax.set_xlabel("Value")
        ax.set_ylabel(depth_col)
        if invert_depth:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title("Logs vs Depth")
        st.pyplot(fig, clear_figure=True)

    else:
        # Plotly WebGL
        import plotly.graph_objects as go
        fig = go.Figure()
        yvals = plot_df[depth_col].to_numpy()
        for c in st.session_state.selected_cols:
            xvals = plot_df[c].to_numpy()
            fig.add_trace(go.Scattergl(
                x=xvals, y=yvals, mode="lines", name=c, line=dict(width=1)
            ))
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title=depth_col,
            height=600,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        if invert_depth:
            fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.divider()
with st.expander("Data preview"):
    st.dataframe(df.head(50))

# ================= Lithology Prediction (Model Bank) =================
st.header("Lithology Prediction")
st.caption("Uses pre-trained models for many feature combinations so you don't have to retrain.")

model_bank_status = st.empty()

try:
    from model_bank import predict_with_best_model, select_best_entry_for_available, load_registry
    registry = load_registry()
    st.caption(f"Loaded model registry with **{len(registry.get('models', []))}** models.")
except Exception as e:
    st.error(f"Model bank not available: {e}")
    registry = None

# Let user choose which columns to use for prediction (defaults to all intersecting known logs)
predict_cols_default = [c for c in LOG_BUTTONS if c in df.columns]
predict_cols = st.multiselect(
    "Columns to use for prediction (we'll auto-pick the best model that fits them)",
    options=[c for c in df.columns if c != depth_col],
    default=predict_cols_default
)

if st.button("Predict Lithology", disabled=registry is None):
    try:
        # We do not drop NaNs here; pipeline will handle via imputers.
        preds, entry = predict_with_best_model(df, predict_cols)
        out_df = df.copy()
        out_df["pred_lithology"] = preds
        st.success(f"Predicted using model trained on features: {entry['required_features']} "
                   f"(F1_macro={entry['metrics'].get('f1_macro', 'N/A'):.3f})")
        st.dataframe(out_df[[depth_col] + entry['required_features'] + ["pred_lithology"]].head(100))

        # Offer CSV download
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV with predictions", data=csv_bytes, file_name="with_pred_lithology.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
