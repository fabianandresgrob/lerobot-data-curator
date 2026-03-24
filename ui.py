"""
ui.py — Interactive results explorer for lerobot-data-curator.

Loads pre-computed technical and semantic score JSONs and provides
interactive visualizations for exploring data quality.

Run with:
    streamlit run ui.py -- --results_dir path/to/results/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LeRobot Data Curator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "GOOD":           "#2ecc71",
    "FAIL_SEMANTIC":  "#e74c3c",
    "FAIL_TECHNICAL": "#f39c12",
    "FAIL_BOTH":      "#8e44ad",
}

KNOWN_CONDITIONS = {
    "j-m-h_pick_place_clean_realsense_downscaled":                "clean",
    "fabiangrob_pick_place_wrong_cube_realsense_downscaled":      "wrong_cube",
    "fabiangrob_pick_place_task_fail_realsense_downscaled":       "task_fail",
    "fabiangrob_pick_place_extra_objects_realsense_downscaled":   "extra_objects",
    "fabiangrob_pick_place_bad_lighting_realsense_downscaled":    "bad_lighting",
    "fabiangrob_pick_place_shakiness_realsense_downscaled":       "shakiness",
    "fabiangrob_pick_place_occluded_top_cam_realsense_downscaled":"occluded_top_cam",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def short_name(path: Path) -> str:
    """Derive a short human-readable label from a JSON filename."""
    stem = path.stem
    for suffix in ("_scores", "_vlmonly", "_full"):
        stem = stem.replace(suffix, "")
    stem = stem.removeprefix("baseline_")
    return KNOWN_CONDITIONS.get(stem, stem)


@st.cache_data
def load_technical(path: str) -> pd.DataFrame:
    """
    Load technical scores from a *_scores.json file.
    Also extracts semantic_score if present (output of curate_dataset.py --semantic).
    """
    with open(path) as f:
        data = json.load(f)
    rows = []
    for entry in data:
        row = {
            "episode_id": entry["episode_id"],
            "camera":     entry.get("camera_type", ""),
            "aggregate":  entry["aggregate_score"],
        }
        if "semantic_score" in entry and entry["semantic_score"] is not None:
            row["semantic_embedded"] = entry["semantic_score"]
        for k, v in entry.get("per_attribute_scores", {}).items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    # Take max aggregate across cameras per episode
    agg = df.groupby("episode_id")["aggregate"].max().reset_index()
    criteria = [c for c in df.columns
                if c not in ("episode_id", "camera", "aggregate", "semantic_embedded")]
    for c in criteria:
        agg[c] = df.groupby("episode_id")[c].mean().values
    # If semantic scores are embedded, carry them over (one per episode, not per camera)
    if "semantic_embedded" in df.columns:
        sem = df.dropna(subset=["semantic_embedded"]).groupby("episode_id")["semantic_embedded"].first()
        agg["semantic_embedded"] = agg["episode_id"].map(sem)
    return agg


@st.cache_data
def load_semantic(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    rows = []
    for entry in data:
        rows.append({
            "episode_id":    entry["episode_idx"],
            "semantic":      entry.get("semantic_score"),
            "predicted":     entry.get("predicted_label"),
            "ground_truth":  entry.get("ground_truth_label"),
            "condition":     entry.get("condition", ""),
            "mode":          entry.get("mode", ""),
        })
    return pd.DataFrame(rows)


def merge_scores(tech_df: pd.DataFrame, sem_df: pd.DataFrame,
                 tech_thresh: float, sem_thresh: float) -> pd.DataFrame:
    sem_cols = ["episode_id", "semantic"]
    if "ground_truth" in sem_df.columns:
        sem_cols.append("ground_truth")
    if "condition" in sem_df.columns:
        sem_cols.append("condition")
    df = pd.merge(tech_df, sem_df[sem_cols], on="episode_id", how="outer")
    p_tech = df["aggregate"] >= tech_thresh
    p_sem  = df["semantic"]  >= sem_thresh

    def status(row):
        t = row["aggregate"] >= tech_thresh if pd.notna(row["aggregate"]) else None
        s = row["semantic"]  >= sem_thresh  if pd.notna(row["semantic"])  else None
        if t is None:  return "FAIL_TECHNICAL"
        if s is None:  return "FAIL_SEMANTIC"
        if t and s:    return "GOOD"
        if t and not s: return "FAIL_SEMANTIC"
        if not t and s: return "FAIL_TECHNICAL"
        return "FAIL_BOTH"

    df["status"] = df.apply(status, axis=1)
    return df

# ── Chart helpers ──────────────────────────────────────────────────────────────

def scatter_2d(df, tech_thresh, sem_thresh, title=""):
    fig = px.scatter(
        df.dropna(subset=["aggregate", "semantic"]),
        x="aggregate",
        y="semantic",
        color="status",
        color_discrete_map=STATUS_COLORS,
        hover_data=["episode_id"],
        title=title or "Technical vs Semantic Score",
        labels={"aggregate": "Technical score (aggregate)", "semantic": "Semantic score"},
    )
    fig.add_vline(x=tech_thresh, line_dash="dash", line_color="gray",
                  annotation_text=f"tech={tech_thresh}")
    fig.add_hline(y=sem_thresh,  line_dash="dash", line_color="gray",
                  annotation_text=f"sem={sem_thresh}")
    fig.update_layout(height=450, legend_title="Status")
    return fig


def status_bar(df):
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig = px.bar(
        counts, x="Status", y="Count",
        color="Status", color_discrete_map=STATUS_COLORS,
        title="Episode counts by status",
    )
    fig.update_layout(height=300, showlegend=False)
    return fig


def score_hist(df, col, thresh, color, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=25,
                               marker_color=color, opacity=0.75, name=col))
    fig.add_vline(x=thresh, line_dash="dash", line_color="red",
                  annotation_text=f"threshold={thresh}")
    fig.update_layout(title=title, xaxis_title="Score",
                      yaxis_title="Episodes", height=300)
    return fig


def criteria_boxplots(df, criteria):
    long = df[["episode_id"] + criteria].melt(
        id_vars="episode_id", var_name="Criterion", value_name="Score"
    )
    fig = px.box(long, x="Criterion", y="Score", points="outliers",
                 title="Technical criteria distribution",
                 color="Criterion")
    fig.update_layout(height=350, showlegend=False)
    return fig

# ── Sidebar: results directory ─────────────────────────────────────────────────

def get_results_dir() -> Path:
    # Allow passing via CLI: streamlit run ui.py -- --results_dir path/
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results_dir", default=None)
    try:
        args, _ = parser.parse_known_args()
        if args.results_dir:
            return Path(args.results_dir)
    except SystemExit:
        pass
    return None


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.title("🤖 LeRobot Data Curator — Results Explorer")
    st.caption("Explore technical and semantic quality scores for your robot episodes.")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data")

        default_dir = str(get_results_dir() or "./results")
        results_dir = Path(st.text_input("Results directory", value=default_dir))

        # Discover available score files
        tech_files, sem_files = {}, {}
        if results_dir.exists():
            for f in sorted(results_dir.glob("*_scores.json")):
                tech_files[short_name(f)] = f
            # Semantic scores from evaluate_semantic_baseline.py (separate files)
            for f in sorted(results_dir.glob("baseline_*_full.json")):
                sem_files[short_name(f)] = f
            # Semantic scores embedded in technical files (from curate_dataset.py --semantic)
            # These are already loaded via load_technical(); mark them available
            for name, f in tech_files.items():
                if name not in sem_files:
                    with open(f) as fh:
                        first = json.load(fh)
                    if first and first[0].get("semantic_score") is not None:
                        sem_files[name] = f  # same file, semantic_embedded flag used

        if not tech_files and not sem_files:
            st.warning("No score files found. Run curate_dataset.py or "
                       "evaluate_semantic_baseline.py first.")
            return

        all_conditions = sorted(set(tech_files) | set(sem_files))
        selected = st.multiselect("Conditions to load", all_conditions,
                                  default=all_conditions)

        st.markdown("---")
        st.header("Thresholds")

        # Initialise shared threshold state
        for k, v in [("tech_thresh", 0.5), ("sem_thresh", 0.5)]:
            if k not in st.session_state:
                st.session_state[k] = v

        def _sync(src, dst, shared):
            st.session_state[shared] = st.session_state[src]
            st.session_state[dst]    = st.session_state[src]

        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            st.slider("Technical threshold", 0.0, 1.0, step=0.01,
                      key="_tech_slider",
                      value=st.session_state["tech_thresh"],
                      on_change=_sync, args=("_tech_slider", "_tech_num", "tech_thresh"))
        with t_col2:
            st.number_input("", 0.0, 1.0, step=0.01, format="%.2f",
                            key="_tech_num",
                            value=st.session_state["tech_thresh"],
                            on_change=_sync, args=("_tech_num", "_tech_slider", "tech_thresh"),
                            label_visibility="hidden")
        tech_thresh = st.session_state["tech_thresh"]

        s_col1, s_col2 = st.columns([3, 1])
        with s_col1:
            st.slider("Semantic threshold", 0.0, 1.0, step=0.01,
                      key="_sem_slider",
                      value=st.session_state["sem_thresh"],
                      on_change=_sync, args=("_sem_slider", "_sem_num", "sem_thresh"))
        with s_col2:
            st.number_input("", 0.0, 1.0, step=0.01, format="%.2f",
                            key="_sem_num",
                            value=st.session_state["sem_thresh"],
                            on_change=_sync, args=("_sem_num", "_sem_slider", "sem_thresh"),
                            label_visibility="hidden")
        sem_thresh = st.session_state["sem_thresh"]

    if not selected:
        st.info("Select at least one condition in the sidebar.")
        return

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_single, tab_compare = st.tabs(["Single condition", "Compare conditions"])

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1 — single condition deep-dive
    # ══════════════════════════════════════════════════════════════════════════
    with tab_single:
        condition = st.selectbox("Condition", selected)

        tech_df = load_technical(str(tech_files[condition])) if condition in tech_files else None
        # Semantic: prefer separate baseline file; fall back to embedded scores
        if condition in sem_files:
            sem_path = sem_files[condition]
            if str(sem_path).endswith("_scores.json") and tech_df is not None and "semantic_embedded" in tech_df.columns:
                sem_df = tech_df[["episode_id", "semantic_embedded"]].rename(columns={"semantic_embedded": "semantic"})
            else:
                sem_df = load_semantic(str(sem_path))
        else:
            sem_df = None

        has_tech = tech_df is not None
        has_sem  = sem_df  is not None

        if not has_tech and not has_sem:
            st.warning("No score files found for this condition.")
            st.stop()

        # Merge into a single per-episode dataframe
        if has_tech and has_sem:
            df = merge_scores(tech_df, sem_df, tech_thresh, sem_thresh)
        elif has_tech:
            df = tech_df.copy()
            df["semantic"] = None
            df["status"] = df["aggregate"].apply(
                lambda s: "GOOD" if s >= tech_thresh else "FAIL_TECHNICAL")
        else:
            df = sem_df[["episode_id", "semantic"]].copy()
            df["aggregate"] = None
            df["status"] = df["semantic"].apply(
                lambda s: "GOOD" if pd.notna(s) and s >= sem_thresh else "FAIL_SEMANTIC")

        # ── Metrics row ───────────────────────────────────────────────────────
        n_total = len(df)
        n_good  = (df["status"] == "GOOD").sum()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total episodes", n_total)
        c2.metric("GOOD",           n_good,
                  f"{n_good/n_total*100:.0f}%")
        c3.metric("FAIL_SEMANTIC",  (df["status"] == "FAIL_SEMANTIC").sum())
        c4.metric("FAIL_TECHNICAL", (df["status"] == "FAIL_TECHNICAL").sum())
        c5.metric("FAIL_BOTH",      (df["status"] == "FAIL_BOTH").sum())

        # ── Filtering quality panel (only when ground truth is available) ──────
        has_gt = "ground_truth" in df.columns and df["ground_truth"].notna().any()
        if has_gt:
            kept = df[df["status"] == "GOOD"]
            removed = df[df["status"] != "GOOD"]

            n_clean_total   = (df["ground_truth"] == 1).sum()
            n_bad_total     = (df["ground_truth"] == 0).sum()
            n_clean_kept    = (kept["ground_truth"] == 1).sum()
            n_clean_removed = (removed["ground_truth"] == 1).sum()
            n_bad_kept      = (kept["ground_truth"] == 0).sum()
            n_bad_removed   = (removed["ground_truth"] == 0).sum()

            clean_retention = n_clean_kept / n_clean_total if n_clean_total > 0 else 0
            bad_recall      = n_bad_removed / n_bad_total  if n_bad_total  > 0 else 0
            purity          = n_clean_kept / len(kept)     if len(kept)    > 0 else 0
            noise_remaining = n_bad_kept

            st.markdown("---")
            st.subheader("Filtering quality (ground truth available)")
            st.caption("Shows how well the current thresholds clean the dataset — only possible because this is a simulated dataset with known labels.")

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Clean episodes kept",
                      f"{n_clean_kept} / {n_clean_total}",
                      f"{clean_retention*100:.1f}%",
                      delta_color="normal")
            q2.metric("Bad episodes removed",
                      f"{n_bad_removed} / {n_bad_total}",
                      f"{bad_recall*100:.1f}%",
                      delta_color="normal")
            q3.metric("Dataset purity after filtering",
                      f"{purity*100:.1f}%",
                      help="Fraction of kept episodes that are actually clean.")
            q4.metric("Noise remaining",
                      noise_remaining,
                      f"{n_bad_kept/len(kept)*100:.1f}% of kept" if len(kept) > 0 else "—",
                      delta_color="inverse")

            # Confusion matrix as a stacked bar
            cm_df = pd.DataFrame({
                "Group":  ["Clean", "Clean", "Bad", "Bad"],
                "Outcome": ["Kept", "Removed", "Kept (noise)", "Removed"],
                "Count":  [n_clean_kept, n_clean_removed, n_bad_kept, n_bad_removed],
            })
            cm_colors = {
                "Kept":           "#2ecc71",
                "Removed":        "#e74c3c",
                "Kept (noise)":   "#f39c12",
            }
            fig_cm = px.bar(
                cm_df, x="Group", y="Count", color="Outcome",
                color_discrete_map=cm_colors,
                title="Filtering outcome by ground truth label",
                text="Count",
            )
            fig_cm.update_traces(textposition="inside")
            fig_cm.update_layout(height=320)

            # Per-condition breakdown of bad episodes kept vs removed
            if "condition" in df.columns:
                bad_df = df[df["ground_truth"] == 0].copy()
                bad_df["outcome"] = bad_df["status"].apply(
                    lambda s: "Removed" if s != "GOOD" else "Kept (noise)")
                cond_counts = (
                    bad_df.groupby(["condition", "outcome"])
                    .size().reset_index(name="count")
                )
                fig_cond = px.bar(
                    cond_counts, x="condition", y="count", color="outcome",
                    color_discrete_map={"Removed": "#2ecc71", "Kept (noise)": "#f39c12"},
                    barmode="stack",
                    title="Bad episodes: removed vs kept, per condition",
                    labels={"count": "Episodes", "condition": "Condition"},
                )
                fig_cond.update_layout(height=320)
                col_cm, col_cond = st.columns(2)
                with col_cm:
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col_cond:
                    st.plotly_chart(fig_cond, use_container_width=True)
            else:
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")

        # ── 2D scatter (main viz) ─────────────────────────────────────────────
        if has_tech and has_sem:
            st.plotly_chart(
                scatter_2d(df, tech_thresh, sem_thresh, f"{condition} — Technical vs Semantic"),
                use_container_width=True,
            )
        elif has_tech:
            st.plotly_chart(
                score_hist(df, "aggregate", tech_thresh, "#3498db",
                           f"{condition} — Technical score distribution"),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                score_hist(df, "semantic", sem_thresh, "#e74c3c",
                           f"{condition} — Semantic score distribution"),
                use_container_width=True,
            )

        # ── Score distributions side by side ──────────────────────────────────
        if has_tech and has_sem:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    score_hist(df, "aggregate", tech_thresh, "#3498db",
                               "Technical score distribution"),
                    use_container_width=True,
                )
            with col_b:
                st.plotly_chart(
                    score_hist(df, "semantic", sem_thresh, "#e74c3c",
                               "Semantic score distribution"),
                    use_container_width=True,
                )

        # ── Status breakdown + criteria ───────────────────────────────────────
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(status_bar(df), use_container_width=True)
        with col_right:
            if has_tech:
                criteria = [c for c in df.columns
                            if c not in ("episode_id", "aggregate", "semantic", "status")]
                if criteria:
                    st.plotly_chart(criteria_boxplots(df, criteria),
                                    use_container_width=True)

        # ── Detailed table ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Per-episode table")
        status_opts = ["All"] + sorted(df["status"].unique().tolist())
        filt = st.selectbox("Filter by status", status_opts, key="single_filter")
        show_df = df if filt == "All" else df[df["status"] == filt]

        float_fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and pd.notna(x) else str(x)
        fmt = {}
        if "aggregate" in show_df.columns:
            fmt["aggregate"] = float_fmt
        if "semantic" in show_df.columns:
            fmt["semantic"] = float_fmt
        numeric_cols = show_df.select_dtypes(include="number").columns
        criteria = [c for c in numeric_cols
                    if c not in ("episode_id", "aggregate", "semantic")]
        fmt.update({c: float_fmt for c in criteria})

        def color_status(val):
            colors = {
                "GOOD": "background-color:#d5f4e6",
                "FAIL_SEMANTIC": "background-color:#fad7d7",
                "FAIL_TECHNICAL": "background-color:#fef3cd",
                "FAIL_BOTH": "background-color:#e8d5f4",
            }
            return colors.get(val, "")

        styled = show_df.style.format(fmt).map(color_status, subset=["status"])
        st.dataframe(styled, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2 — compare conditions
    # ══════════════════════════════════════════════════════════════════════════
    with tab_compare:
        st.subheader("Compare score distributions across conditions")

        # Build combined dataframe
        all_dfs = []
        for cond in selected:
            t = load_technical(str(tech_files[cond])) if cond in tech_files else None
            if cond in sem_files:
                sem_path = sem_files[cond]
                if str(sem_path).endswith("_scores.json") and t is not None and "semantic_embedded" in t.columns:
                    s = t[["episode_id", "semantic_embedded"]].rename(columns={"semantic_embedded": "semantic"})
                else:
                    s = load_semantic(str(sem_path))
            else:
                s = None
            if t is not None and s is not None:
                df = merge_scores(t, s, tech_thresh, sem_thresh)
            elif t is not None:
                df = t.copy(); df["semantic"] = None
                df["status"] = df["aggregate"].apply(
                    lambda v: "GOOD" if v >= tech_thresh else "FAIL_TECHNICAL")
            elif s is not None:
                df = s[["episode_id", "semantic"]].copy(); df["aggregate"] = None
                df["status"] = df["semantic"].apply(
                    lambda v: "GOOD" if pd.notna(v) and v >= sem_thresh else "FAIL_SEMANTIC")
            else:
                continue
            df["condition"] = cond
            all_dfs.append(df)

        if not all_dfs:
            st.info("No data loaded.")
            st.stop()

        combined = pd.concat(all_dfs, ignore_index=True)

        # ── Aggregate score distribution ───────────────────────────────────────
        if combined["aggregate"].notna().any():
            st.markdown("**Technical score distribution**")
            fig = px.histogram(
                combined.dropna(subset=["aggregate"]),
                x="aggregate", color="condition",
                nbins=30, opacity=0.6, barmode="overlay",
                labels={"aggregate": "Technical score (aggregate)"},
            )
            fig.add_vline(x=tech_thresh, line_dash="dash", line_color="black",
                          annotation_text=f"threshold={tech_thresh}")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Technical score CDF**")
            st.caption("Curves sitting lower = more episodes with higher scores = better quality.")
            fig_cdf = px.ecdf(
                combined.dropna(subset=["aggregate"]),
                x="aggregate", color="condition",
                labels={"aggregate": "Technical score (aggregate)"},
            )
            fig_cdf.add_vline(x=tech_thresh, line_dash="dash", line_color="black")
            fig_cdf.update_layout(yaxis_title="Cumulative fraction (≤ score)")
            st.plotly_chart(fig_cdf, use_container_width=True)

        # ── Semantic score distribution ────────────────────────────────────────
        if combined["semantic"].notna().any():
            st.markdown("**Semantic score distribution**")
            fig_sem = px.histogram(
                combined.dropna(subset=["semantic"]),
                x="semantic", color="condition",
                nbins=30, opacity=0.6, barmode="overlay",
                labels={"semantic": "Semantic score"},
            )
            fig_sem.add_vline(x=sem_thresh, line_dash="dash", line_color="black",
                              annotation_text=f"threshold={sem_thresh}")
            st.plotly_chart(fig_sem, use_container_width=True)

        # ── Status breakdown per condition ─────────────────────────────────────
        st.markdown("**Filtering outcome by condition**")
        st.caption("At the current threshold settings, how many episodes fall into each status.")
        status_counts = (
            combined.groupby(["condition", "status"])
            .size().reset_index(name="count")
        )
        fig_status = px.bar(
            status_counts, x="condition", y="count",
            color="status", color_discrete_map=STATUS_COLORS,
            barmode="stack",
            labels={"count": "Episodes", "condition": "Condition"},
        )
        fig_status.update_layout(height=400)
        st.plotly_chart(fig_status, use_container_width=True)

        # ── 2D scatter coloured by condition ───────────────────────────────────
        if combined["aggregate"].notna().any() and combined["semantic"].notna().any():
            st.markdown("**Technical vs Semantic — all conditions**")
            fig_2d = px.scatter(
                combined.dropna(subset=["aggregate", "semantic"]),
                x="aggregate", y="semantic",
                color="condition",
                hover_data=["episode_id", "status"],
                labels={"aggregate": "Technical score", "semantic": "Semantic score"},
            )
            fig_2d.add_vline(x=tech_thresh, line_dash="dash", line_color="gray")
            fig_2d.add_hline(y=sem_thresh,  line_dash="dash", line_color="gray")
            fig_2d.update_layout(height=500)
            st.plotly_chart(fig_2d, use_container_width=True)

        # ── Summary table ──────────────────────────────────────────────────────
        st.markdown("**Summary statistics**")
        summary_rows = []
        for cond in selected:
            sub = combined[combined["condition"] == cond]
            n = len(sub)
            row = {"condition": cond, "n_episodes": n}
            if sub["aggregate"].notna().any():
                row["tech_mean"]   = sub["aggregate"].mean()
                row["tech_median"] = sub["aggregate"].median()
            if sub["semantic"].notna().any():
                row["sem_mean"]    = sub["semantic"].mean()
                row["sem_median"]  = sub["semantic"].median()
            for s in STATUS_COLORS:
                row[s] = (sub["status"] == s).sum()
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        fmt = {c: "{:.3f}" for c in ("tech_mean", "tech_median", "sem_mean", "sem_median")
               if c in summary_df.columns}
        st.dataframe(summary_df.style.format(fmt), use_container_width=True)


if __name__ == "__main__":
    main()
