"""iStrain: strain management dashboards for rodents and aquatic models (Streamlit).

Run locally:   streamlit run app_scratch.py
Data:          breeding_pairs.csv + litters.csv, looked up in ./data (rodents) and ./data/aquatic
               (aquatic), ../../data, or $STRAIN_DATA_DIR. Other files can be uploaded from the sidebar.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

WEAN_AGE_DAYS = 21
MIN_STRAIN_MONTHS = 6   # strains with a shorter record are kept out of the regression plots
DEFAULT_BENCHMARKS = {"BALB/cJ": 6.0, "C57BL/6J": 7.0, "FVB/NJ": 8.0, "C3H": 6.0,
                      "L404": 6.0, "L438": 7.0, "L134": 8.0, "L127": 6.0}
PLATFORMS = {
    "rodent": dict(
        label="Rodents", icon="🐭", title="Mouse strain management", data_sub="",
        unit="pair", units="pairs", id_col="pair_id", schedule=True,
        t=dict(
            batch="litter", batches="litters", batch_cap="Litter", young="pups", y_lbl="Pups per litter",
            per="pups/litter", wean_tab="Weaning", date_lbl="Date of birth", range_lbl="Litter DOB range",
            bench_lbl="Benchmark litter size (pups)", age_cap=f"Standard weaning age: {WEAN_AGE_DAYS} days",
            kpi="Litters to wean", kpi_help="Litters with no weaning data recorded yet "
                                            "(pre-weaning or past the expected date).",
            surv_title="Probability of litter survival to weaning: logistic-regression estimate",
            surv_y="P(pup survives to weaning)",
            surv_msg="Per strain, logistic regression on weaned litters: success = pup weaned, failure = pup born "
                     "but not weaned (litter size minus weaned). ",
            sex_on="weaned litters", last_lbl="last litter", next_lbl="Next litter", dob_short="DOB",
            id_lbl="litter_id", size_lbl="litter_size", status_map={},
        )),
    "aquatic": dict(
        label="Aquatic", icon="🐟", title="Aquatic strain management", data_sub="aquatic",
        unit="tank", units="tanks", id_col="tank_id", schedule=False,
        t=dict(
            batch="clutch", batches="clutches", batch_cap="Clutch", young="embryos", y_lbl="Embryos per clutch",
            per="embryos/clutch", wean_tab="Juvenile survival", date_lbl="Spawning date",
            range_lbl="Clutch spawning-date range", bench_lbl="Benchmark clutch size (embryos)",
            age_cap=f"Grow-out to juvenile stage: {WEAN_AGE_DAYS} dpf",
            kpi="Clutches in grow-out", kpi_help="Clutches that have not yet reached the juvenile stage "
                                                 f"({WEAN_AGE_DAYS} dpf) or have no survivor count recorded.",
            surv_title="Probability of clutch survival to the juvenile stage (21 dpf): logistic-regression estimate",
            surv_y="P(embryo survives to 21 dpf)",
            surv_msg="Per strain, logistic regression on clutches that reached 21 dpf: success = embryo survived to "
                     "21 dpf, failure = embryo spawned but not surviving (clutch size minus survivors). ",
            sex_on="clutches that reached 21 dpf", last_lbl="last clutch", next_lbl="Next clutch",
            dob_short="spawn date", id_lbl="clutch_id", size_lbl="clutch_size",
            status_map={"Weaned": "At 21 dpf", "Pre-weaning": "In grow-out",
                        "Weaning due / overdue": "Past 21 dpf, not recorded"},
        )),
}
# Okabe-Ito, colour-blind safe; one fixed colour per strain across every chart
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]
RECENT = "Last six months"
PERIODS = {
    RECENT: pd.DateOffset(months=6),
    "Last year": pd.DateOffset(years=1),
    "All records": None,
}
PAIR_COLS = {"pair_id", "strain", "male_id", "female_id", "setup_date", "status"}
LITTER_COLS = {
    "litter_id", "pair_id", "strain", "dob", "litter_size",
    "wean_date", "n_males_weaned", "n_females_weaned",
}

st.set_page_config(page_title="iStrain", page_icon="🧬", layout="wide")


# --------------------------------------------------------------------------- data
def _find_data_dir(sub: str = "") -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("STRAIN_DATA_DIR"), here / "data", here.parent.parent / "data", Path.cwd() / "data"]
    for c in candidates:
        if c and (Path(c) / sub / "litters.csv").exists():
            return Path(c) / sub
    return None


@st.cache_data(show_spinner=False)
def _read_csv(src, mtime: float = 0.0) -> pd.DataFrame:
    """`mtime` is part of the cache key, so editing a file on disk invalidates the cached copy."""
    return pd.read_csv(src)


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        st.error(f"{name} is missing column(s): {', '.join(sorted(missing))}")
        st.stop()


def load_data(sub: str, platform: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    with st.sidebar.expander("Data source"):
        up_p = st.file_uploader("breeding_pairs.csv", type="csv", key=f"up_pairs_{platform}")
        up_l = st.file_uploader("litters.csv", type="csv", key=f"up_litters_{platform}")
    if up_p is not None and up_l is not None:
        pairs, litters, source = _read_csv(up_p), _read_csv(up_l), "uploaded files"
    else:
        d = _find_data_dir(sub)
        if d is None:
            st.error("No data found. Upload breeding_pairs.csv and litters.csv in the sidebar.")
            st.stop()
        fp, fl = d / "breeding_pairs.csv", d / "litters.csv"
        pairs, litters = _read_csv(fp, fp.stat().st_mtime), _read_csv(fl, fl.stat().st_mtime)
        source = str(d)
    _check_columns(pairs, PAIR_COLS, "breeding_pairs.csv")
    _check_columns(litters, LITTER_COLS, "litters.csv")
    if "sacrifice_date" not in pairs.columns:       # older files: every pair is still breeding
        pairs = pairs.assign(sacrifice_date=pd.NaT)
    pairs = pairs.assign(
        setup_date=pd.to_datetime(pairs["setup_date"], errors="coerce"),
        sacrifice_date=pd.to_datetime(pairs["sacrifice_date"], errors="coerce"),
    )
    litters = litters.assign(
        dob=pd.to_datetime(litters["dob"], errors="coerce"),
        wean_date=pd.to_datetime(litters["wean_date"], errors="coerce"),
        litter_size=pd.to_numeric(litters["litter_size"], errors="coerce"),
        n_males_weaned=pd.to_numeric(litters["n_males_weaned"], errors="coerce"),
        n_females_weaned=pd.to_numeric(litters["n_females_weaned"], errors="coerce"),
    )
    return pairs, litters, source


def add_pair_state(pairs: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """A pair is active on `as_of` if it was set up and not yet sacrificed."""
    started = pairs["setup_date"] <= as_of
    gone = pairs["sacrifice_date"].notna() & (pairs["sacrifice_date"] <= as_of)
    state = np.where(~started, "not set up", np.where(gone, "sacrificed", "active"))
    return pairs.assign(state=state)


def enrich_litters(litters: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Add weaning status and derived columns. Blank weaning data is 'not weaned yet'."""
    df = litters.copy()
    df["weaned_recorded"] = df["n_males_weaned"].notna() & df["n_females_weaned"].notna()
    df["n_weaned"] = df["n_males_weaned"].fillna(0) + df["n_females_weaned"].fillna(0)
    df.loc[~df["weaned_recorded"], "n_weaned"] = pd.NA
    df["wean_survival"] = (df["n_weaned"].astype("float") / df["litter_size"]).where(df["weaned_recorded"])
    df["expected_wean"] = df["wean_date"].fillna(df["dob"] + pd.Timedelta(days=WEAN_AGE_DAYS))
    days_to = (df["expected_wean"] - as_of).dt.days
    df["days_to_wean"] = days_to.where(~df["weaned_recorded"])
    df["status"] = "Weaned"
    df.loc[~df["weaned_recorded"] & (days_to > 0), "status"] = "Pre-weaning"
    df.loc[~df["weaned_recorded"] & (days_to <= 0), "status"] = "Weaning due / overdue"
    return df


def pair_summary(pairs: pd.DataFrame, lit: pd.DataFrame, as_of: pd.Timestamp, bench: dict) -> pd.DataFrame:
    rows = []
    for p in pairs.itertuples():
        g = lit[lit["pair_id"] == p.pair_id].sort_values("dob")
        intervals = g["dob"].diff().dt.days.dropna()
        last = g["dob"].max() if len(g) else pd.NaT
        mean_int = intervals.mean() if len(intervals) else float("nan")
        nxt = pd.NaT
        if p.state == "active" and pd.notna(last) and pd.notna(mean_int):   # no next litter for sacrificed pairs
            nxt = last + pd.Timedelta(days=round(mean_int))
        rows.append({
            "pair_id": p.pair_id, "strain": p.strain, "state": p.state,
            "setup_date": p.setup_date, "sacrifice_date": p.sacrifice_date,
            "litters": len(g), "pups_born": g["litter_size"].sum(),
            "mean_litter_size": g["litter_size"].mean(),
            "benchmark": bench.get(p.strain, float("nan")),
            "mean_interval_d": mean_int,
            "last_litter": last,
            "days_since_last": (as_of - last).days if pd.notna(last) else pd.NA,
            "expected_next": nxt,
            "days_overdue": max((as_of - nxt).days, 0) if pd.notna(nxt) else pd.NA,
        })
    out = pd.DataFrame(rows)
    out["vs_benchmark"] = out["mean_litter_size"] - out["benchmark"]
    return out


def seasonal_trend_fit(y: np.ndarray, quarter: np.ndarray, t: np.ndarray, level: float = 0.95):
    """OLS of y on a linear time trend plus quarter-of-year effects (Q1 = reference).

    `t` is time in trimesters. Returns the adjusted mean for Q1..Q4 at the mid-point of the record,
    with its CI, and the time slope (per trimester) with its CI. None when there is too little data.
    """
    n = len(y)
    if n < 10 or len(set(quarter)) < 4:
        return None
    X = np.column_stack([np.ones(n), t - t.mean()] + [(quarter == q).astype(float) for q in (2, 3, 4)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = n - X.shape[1]
    if df < 1:
        return None
    s2 = ((y - X @ beta) ** 2).sum() / df
    cov = s2 * np.linalg.inv(X.T @ X)
    tcrit = stats.t.ppf(0.5 + level / 2, df)
    est, half = [], []
    for q in (1, 2, 3, 4):
        x0 = np.array([1, 0] + [float(q == k) for k in (2, 3, 4)])
        est.append(x0 @ beta)
        half.append(tcrit * np.sqrt(x0 @ cov @ x0))
    se_slope = np.sqrt(cov[1, 1])
    return np.array(est), np.array(half), beta[1], (beta[1] - tcrit * se_slope, beta[1] + tcrit * se_slope)


def _design(quarter: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(t)), t - t.mean()] + [(quarter == q).astype(float) for q in (2, 3, 4)])


def _mid_point_rows() -> list[np.ndarray]:
    """Design rows for Q1..Q4 at the mid-point of the record (centred time = 0)."""
    return [np.array([1, 0] + [float(q == k) for k in (2, 3, 4)]) for q in (1, 2, 3, 4)]


def logistic_seasonal_fit(succ: np.ndarray, fail: np.ndarray, quarter: np.ndarray, t: np.ndarray,
                          level: float = 0.95):
    """Binomial GLM (logit link) of succ/(succ+fail) on a linear time trend + quarter-of-year effects.

    Fitted by IRLS on litter-level counts; the covariance is scaled by the Pearson dispersion (>= 1)
    because pups of one litter are not independent. Returns P(success) for Q1..Q4 at the mid-point
    of the record with a Wald CI built on the logit scale, plus the odds ratio per trimester with its CI.
    None when there is too little data.
    """
    tot = succ + fail
    keep = tot > 0
    succ, tot, quarter, t = succ[keep], tot[keep], quarter[keep], t[keep]
    n = len(succ)
    if n < 10 or len(set(quarter)) < 4 or succ.sum() == 0 or succ.sum() == tot.sum():
        return None
    X = _design(quarter, t)
    p_par = X.shape[1]
    beta = np.zeros(p_par)
    for _ in range(50):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(tot * mu * (1 - mu), 1e-9)
        z = eta + (succ - tot * mu) / w
        new = np.linalg.solve((X.T * w) @ X + 1e-9 * np.eye(p_par), (X.T * w) @ z)
        done = np.max(np.abs(new - beta)) < 1e-8
        beta = new
        if done:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    w = np.maximum(tot * mu * (1 - mu), 1e-9)
    phi = max(1.0, float((((succ - tot * mu) ** 2) / w).sum() / max(n - p_par, 1)))
    cov = phi * np.linalg.inv((X.T * w) @ X)
    zc = stats.norm.ppf(0.5 + level / 2)
    est, lo, hi = [], [], []
    for x0 in _mid_point_rows():
        eta, se = x0 @ beta, np.sqrt(x0 @ cov @ x0)
        est.append(1 / (1 + np.exp(-eta)))
        lo.append(1 / (1 + np.exp(-(eta - zc * se))))
        hi.append(1 / (1 + np.exp(-(eta + zc * se))))
    se_b = np.sqrt(cov[1, 1])
    return (np.array(est), np.array(lo), np.array(hi), float(np.exp(beta[1])),
            (float(np.exp(beta[1] - zc * se_b)), float(np.exp(beta[1] + zc * se_b))), float(phi))


def logistic_trend_figure(df: pd.DataFrame, succ_col: str, fail_col: str, fit_strains: list[str],
                          t0: pd.Timestamp, ylabel: str, ref_line: float | None = None):
    """One point + error bar per strain and quarter of the year; returns (figure, trend notes, unfitted)."""
    fig = go.Figure()
    notes, unfitted = [], []
    for i, strain in enumerate(fit_strains):
        g = df[df["strain"] == strain]
        res = logistic_seasonal_fit(
            g[succ_col].to_numpy(float), g[fail_col].to_numpy(float), g["dob"].dt.quarter.to_numpy(),
            ((g["dob"] - t0).dt.days / 91.3125).to_numpy())
        if res is None:
            unfitted.append(strain)
            continue
        est, lo, hi, odds, odds_ci, phi = res
        x = np.arange(1, 5) + (i - (len(fit_strains) - 1) / 2) * 0.12
        fig.add_trace(go.Scatter(
            x=x, y=est, mode="lines+markers", name=strain, marker=dict(size=10, color=color_map[strain]),
            line=dict(color=color_map[strain], width=2),
            error_y=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo,
                         color=color_map[strain], thickness=2, width=5),
            hovertemplate=strain + "<br>%{y:.1%}<extra></extra>"))
        notes.append(f"**{strain}** OR {odds:.2f} per trimester (95% CI {odds_ci[0]:.2f} to {odds_ci[1]:.2f})")
    if ref_line is not None:
        fig.add_hline(y=ref_line, line_dash="dot", line_color="#666")
    fig.update_layout(height=360, margin=dict(t=10), legend_title_text="", yaxis_title=ylabel,
                      yaxis=dict(tickformat=".0%", range=[0, 1]),
                      xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4], ticktext=["Q1", "Q2", "Q3", "Q4"],
                                 range=[0.5, 4.5], title=""))
    return fig, notes, unfitted


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion k/n with its Wilson 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, centre - half, centre + half


# --------------------------------------------------------------------------- landing
def landing() -> None:
    st.markdown(
        """
        <style>
        div.stButton > button { height: 8rem; border-radius: 1rem; }
        div.stButton > button p { font-size: 2.2rem; }
        </style>
        """, unsafe_allow_html=True)
    st.write("")
    st.markdown("<h1 style='text-align:center'>Welcome to the iStrain - the strain manager of your model organism</h1>",
                unsafe_allow_html=True)
    st.write("")
    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    for col, key in ((c1, "rodent"), (c2, "aquatic")):
        cfg_ = PLATFORMS[key]
        if col.button(f"{cfg_['icon']}  {cfg_['label']}", key=f"go_{key}", width="stretch"):
            st.session_state["platform"] = key
            st.rerun()


platform = st.session_state.get("platform")
if platform not in PLATFORMS:
    landing()
    st.stop()
cfg = PLATFORMS[platform]
T = cfg["t"]
if st.sidebar.button("← iStrain home", key="home"):
    st.session_state["platform"] = None
    st.rerun()

# --------------------------------------------------------------------------- UI
pairs_raw, litters_raw, source = load_data(cfg["data_sub"], platform)
strains = sorted(pairs_raw["strain"].unique())
color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(strains)}

st.sidebar.title(f"{cfg['icon']} Strain management")
sel_strains = st.sidebar.multiselect("Strains", strains, default=strains, key=f"{platform}_strains")
as_of = pd.Timestamp(st.sidebar.date_input("As-of date", value=date.today(), key=f"{platform}_asof"))
min_d, max_d = litters_raw["dob"].min().date(), max(litters_raw["dob"].max().date(), as_of.date())
d_from, d_to = st.sidebar.slider(T["range_lbl"], min_d, max_d, (min_d, max_d), key=f"{platform}_dob")
with st.sidebar.expander(T["bench_lbl"]):
    bench = {s: st.number_input(s, 0.0, 20.0, DEFAULT_BENCHMARKS.get(s, 7.0), 0.5, key=f"{platform}_bench_{s}")
             for s in strains}
st.sidebar.caption(f"{T['age_cap']} · Source: {source}")

pairs_raw = add_pair_state(pairs_raw, as_of)
pairs = pairs_raw[pairs_raw["strain"].isin(sel_strains)]
active_pairs = pairs[pairs["state"] == "active"]
# a strain is "established" once its first pair has been set up for MIN_STRAIN_MONTHS
first_setup = pairs_raw.groupby("strain")["setup_date"].min()
established = {s: as_of >= first_setup[s] + pd.DateOffset(months=MIN_STRAIN_MONTHS) for s in strains}
young = [s for s in strains if not established[s]]

lit_all = enrich_litters(litters_raw, as_of)
lit_all = lit_all[lit_all["strain"].isin(sel_strains) & (lit_all["dob"] <= as_of)]
lit = lit_all[(lit_all["dob"].dt.date >= d_from) & (lit_all["dob"].dt.date <= d_to)]
psum = pair_summary(pairs, lit_all, as_of, bench)

st.title(cfg["title"])
if lit.empty:
    st.warning(f"No {T['batches']} match the current filters.")
    st.stop()

tab_names = ["Overview", T["wean_tab"], "Sex ratio"] + (["Weaning schedule"] if cfg["schedule"] else []) \
    + [f"Breeding {cfg['units']}"]
_tabs = st.tabs(tab_names)
tab_over, tab_wean, tab_sex = _tabs[:3]
tab_sched = _tabs[3] if cfg["schedule"] else None
tab_pairs = _tabs[-1]

# ---- Overview
with tab_over:
    k = st.columns(2)
    k[0].metric(f"Active {cfg['units']}", len(active_pairs))
    k[1].metric(T["kpi"], int((~lit["weaned_recorded"]).sum()), help=T["kpi_help"])

    head, sel = st.columns([3, 2], vertical_alignment="bottom")
    head.subheader(f"{T['batch_cap']} size by strain")
    period = sel.selectbox("Period", list(PERIODS), index=2, label_visibility="collapsed", key=f"{platform}_period")
    offset = PERIODS[period]
    lit_p = lit if offset is None else lit[lit["dob"] >= as_of - offset]
    if lit_p.empty:
        st.info(f"No {T['batches']} in this period.")
    else:
        fig = px.box(lit_p, x="strain", y="litter_size", color="strain", points="all",
                     color_discrete_map=color_map, category_orders={"strain": strains},
                     labels={"litter_size": T["y_lbl"], "strain": ""})
        shown = [s for s in strains if s in set(lit_p["strain"])]
        fig.add_trace(go.Scatter(x=shown, y=[bench[s] for s in shown], mode="markers", name="Benchmark",
                                 marker=dict(symbol="line-ew", size=40, line=dict(width=3, color="#444"))))
        fig.update_layout(showlegend=False, height=380, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(lit_p)} {T['batches']} · black bars = strain benchmark")

    st.subheader(f"{T['y_lbl']} by trimester: linear-regression estimate")
    trend_strains = [s for s in strains if s in sel_strains and established[s]]
    excluded = [s for s in strains if s in sel_strains and not established[s]]
    t0 = lit["dob"].min()
    fig = go.Figure()
    trends, too_short = [], []
    for i, strain in enumerate(trend_strains):
        g = lit[lit["strain"] == strain]
        res = seasonal_trend_fit(g["litter_size"].to_numpy(float), g["dob"].dt.quarter.to_numpy(),
                                 ((g["dob"] - t0).dt.days / 91.3125).to_numpy())
        if res is None:
            too_short.append(strain)
            continue
        est, half, slope, slope_ci = res
        x = np.arange(1, 5) + (i - (len(trend_strains) - 1) / 2) * 0.12      # dodge strains within a quarter
        fig.add_trace(go.Scatter(
            x=x, y=est, mode="lines+markers", name=strain, marker=dict(size=10, color=color_map[strain]),
            line=dict(color=color_map[strain], width=2),
            error_y=dict(type="data", array=half, color=color_map[strain], thickness=2, width=5),
            hovertemplate=strain + f"<br>%{{y:.2f}} {T['per']}<extra></extra>"))
        trends.append(f"**{strain}** {slope:+.2f} {T['per']} per trimester (95% CI {slope_ci[0]:+.2f} to {slope_ci[1]:+.2f})")
    fig.add_hline(y=0, line_color="#666", line_width=1.5)
    fig.update_layout(height=380, margin=dict(t=10), legend_title_text="", yaxis_title=T["y_lbl"],
                      xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4], ticktext=["Q1", "Q2", "Q3", "Q4"],
                                 range=[0.5, 4.5], title=""))
    st.plotly_chart(fig, width="stretch")
    msg = (f"Per strain, OLS on individual {T['batches']} ({T['batch']} size) with a linear time trend plus a "
           f"quarter-of-year effect; points = estimated {T['y_lbl'].lower()} in Q1-Q4 at the mid-point of the "
           "record, pooled across years, error bars = 95% CI. ")
    if trends:
        msg += "Time trend: " + " · ".join(trends) + ". "
    if excluded:
        msg += f"Not shown (under {MIN_STRAIN_MONTHS} months of records): {', '.join(excluded)}. "
    if too_short:
        msg += f"Too few {T['batches']} to fit: {', '.join(too_short)}."
    st.caption(msg)

# ---- Weaning
with tab_wean:
    weaned = lit[lit["weaned_recorded"]]
    fit_s = [x for x in strains if x in sel_strains and established[x]]
    skipped = [x for x in strains if x in sel_strains and not established[x]]
    st.subheader(T["surv_title"])
    w = weaned.assign(n_lost=lambda d: d["litter_size"] - d["n_weaned"])
    fig, notes, unfitted = logistic_trend_figure(w, "n_weaned", "n_lost", fit_s, lit["dob"].min(),
                                                 T["surv_y"])
    st.plotly_chart(fig, width="stretch")
    msg = (T["surv_msg"] + "Terms: linear time trend + quarter-of-year effect. Points = "
           "estimated survival in Q1-Q4 at the mid-point of the record, pooled across years; error bars = 95% CI; "
           f"SE inflated for within-{T['batch']} correlation when the data are overdispersed. ")
    if notes:
        msg += "Time trend (odds ratio, 1 = no change): " + " · ".join(notes) + ". "
    if skipped:
        msg += f"Not shown (under {MIN_STRAIN_MONTHS} months of records): {', '.join(skipped)}. "
    if unfitted:
        msg += f"Too little data to fit: {', '.join(unfitted)}."
    st.caption(msg)

# ---- Sex ratio
with tab_sex:
    weaned = lit[lit["weaned_recorded"]]
    fit_s = [x for x in strains if x in sel_strains and established[x]]
    skipped = [x for x in strains if x in sel_strains and not established[x]]
    st.subheader("P(male): logistic-regression estimate")
    sx = weaned[weaned["n_weaned"].astype(float) > 0]
    fig, notes, unfitted = logistic_trend_figure(sx, "n_males_weaned", "n_females_weaned", fit_s,
                                                 lit["dob"].min(), "P(male)", ref_line=0.5)
    st.plotly_chart(fig, width="stretch")
    msg = (f"**Male = success, female = failure.** Per strain, logistic regression on {T['sex_on']} "
           "(linear time trend + quarter-of-year effect); points = estimated P(male) in Q1-Q4 at the "
           "mid-point of the record, error bars = 95% CI, dotted line = 50%. ")
    if notes:
        msg += "Time trend (odds ratio, 1 = no change): " + " · ".join(notes) + ". "
    if skipped:
        msg += f"Not shown (under {MIN_STRAIN_MONTHS} months of records): {', '.join(skipped)}. "
    if unfitted:
        msg += f"Too little data to fit: {', '.join(unfitted)}."
    st.caption(msg)

# ---- Weaning schedule
if tab_sched is not None:
    with tab_sched:
        pick = st.selectbox("Strain", ["All strains"] + [x for x in strains if x in sel_strains], key=f"{platform}_sched_strain")
        pending = lit[~lit["weaned_recorded"]]
        if pick != "All strains":
            pending = pending[pending["strain"] == pick]
        pending = pending.sort_values("expected_wean")
        due = pending[pending["status"] == "Weaning due / overdue"]
        if len(due):
            st.warning(f"{len(due)} litter(s) past the expected weaning date with no weaning data recorded.")
        st.subheader("Weaning schedule")
        if pending.empty:
            st.info("Every litter in range has weaning data.")
        else:
            st.dataframe(
                pending[["litter_id", "pair_id", "strain", "dob", "litter_size", "expected_wean", "days_to_wean", "status"]],
                hide_index=True, width="stretch",
                column_config={
                    "dob": st.column_config.DateColumn("DOB"),
                    "expected_wean": st.column_config.DateColumn("wean on"),
                    "days_to_wean": st.column_config.NumberColumn("days to wean", format="%d"),
                    "litter_size": st.column_config.NumberColumn("pups", format="%d"),
                },
            )
            st.caption(f"Expected weaning = DOB + {WEAN_AGE_DAYS} days.")

# ---- Breeding pairs (active only)
with tab_pairs:
    pick = st.selectbox("Strain", ["All strains"] + [x for x in strains if x in sel_strains], key=f"{platform}_pairs_strain")
    act = psum[psum["state"] == "active"]
    if pick != "All strains":
        act = act[act["strain"] == pick]
    st.subheader(f"Active breeding {cfg['units']} ({len(act)})")
    st.dataframe(
        act.drop(columns=["state", "sacrifice_date"]).sort_values(["strain", "setup_date", "pair_id"])
            .rename(columns={"pair_id": cfg["id_col"]}),
        hide_index=True, width="stretch",
        column_config={
            "setup_date": st.column_config.DateColumn("set up"),
            "litters": st.column_config.NumberColumn(T["batches"], format="%d"),
            "pups_born": st.column_config.NumberColumn(T["young"], format="%d"),
            "mean_litter_size": st.column_config.NumberColumn("mean size", format="%.1f"),
            "benchmark": st.column_config.NumberColumn("bench.", format="%.1f"),
            "vs_benchmark": st.column_config.NumberColumn("Δ bench.", format="%+.1f"),
            "mean_interval_d": st.column_config.NumberColumn("interval (d)", format="%.0f"),
            "last_litter": st.column_config.DateColumn(T["last_lbl"]),
            "days_since_last": st.column_config.NumberColumn("days since", format="%d"),
            "expected_next": st.column_config.DateColumn("next expected"),
            "days_overdue": st.column_config.NumberColumn("overdue (d)", format="%d"),
        },
    )
    st.caption(f"{T['next_lbl']} is estimated as last {T['dob_short']} + the {cfg['unit']}'s own mean interval "
               f"between {T['batches']} (needs at least two). Δ bench. compares the {cfg['unit']}'s mean "
               f"{T['batch']} size to its strain benchmark.")

    tl = lit[lit["pair_id"].isin(act["pair_id"])].sort_values(["strain", "pair_id", "dob"])
    tl = tl.assign(status=tl["status"].map(T["status_map"]).fillna(tl["status"]))
    if not tl.empty:
        st.subheader(f"{T['batch_cap']} timeline")
        order = list(act.sort_values(["strain", "setup_date", "pair_id"])["pair_id"])
        fig = px.scatter(tl, x="dob", y="pair_id", size="litter_size", color="strain",
                         color_discrete_map=color_map, category_orders={"pair_id": order[::-1]},
                         hover_data=["litter_id", "litter_size", "status"], size_max=14,
                         labels={"dob": T["date_lbl"], "pair_id": "", "litter_id": T["id_lbl"], "litter_size": T["size_lbl"]})
        nxt = act.dropna(subset=["expected_next"])
        fig.add_trace(go.Scatter(x=nxt["expected_next"], y=nxt["pair_id"], mode="markers", name="Next expected",
                                 marker=dict(symbol="diamond-open", size=11, line=dict(width=2, color="#444"))))
        fig.add_vline(x=as_of, line_dash="dot", line_color="#888")
        fig.update_layout(height=max(360, 22 * len(order) + 140), margin=dict(t=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Marker size = {T['batch']} size. Dotted line = as-of date; open diamonds = estimated "
                   f"next {T['batch']}.")
