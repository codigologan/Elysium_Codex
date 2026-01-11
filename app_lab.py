import os
import glob
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="LAD Lab Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Erro lendo {path}: {e}")
        return None


def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def _record_error(context: str, err: Exception) -> None:
    st.session_state.setdefault("last_error", {})
    st.session_state["last_error"] = {
        "context": context,
        "error": str(err),
        "traceback": traceback.format_exc(),
    }


def safe_block(context: str):
    """
    Context manager simples: use com 'with safe_block("..."):'.
    Se der erro, registra em session_state e nao derruba o app todo.
    """
    class _Safe:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc is None:
                return False
            _record_error(context, exc)
            st.error(f"Erro em: {context}")
            st.code(st.session_state["last_error"]["traceback"])
            return True

    return _Safe()


def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Falha ao ler CSV: {path} ({e})")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def discover_rl_history_cached(root_runs_dir: str = "runs") -> pd.DataFrame:
    pattern = os.path.join(root_runs_dir, "*", "rl_history.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        run_name = os.path.basename(os.path.dirname(f))
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df["run_name"] = run_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    rename_map = {}
    if "ep" in out.columns and "episode" not in out.columns:
        rename_map["ep"] = "episode"
    if "rewards" in out.columns and "reward" not in out.columns:
        rename_map["rewards"] = "reward"
    if "mean_reward_50" in out.columns and "mean_reward" not in out.columns:
        rename_map["mean_reward_50"] = "mean_reward"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "episode" in out.columns:
        out["episode"] = pd.to_numeric(out["episode"], errors="coerce")
        out = out.dropna(subset=["episode"]).sort_values(["run_name", "episode"])
        out["episode"] = out["episode"].astype(int)

    return out


def discover_rl_history(root_runs_dir: str = "runs") -> pd.DataFrame:
    return discover_rl_history_cached(root_runs_dir)


def add_rolling(df: pd.DataFrame, value_col: str, window: int) -> pd.DataFrame:
    if value_col not in df.columns:
        return df
    df = df.copy()
    df[f"{value_col}_ma"] = (
        df.groupby("run_name")[value_col]
          .rolling(window=window, min_periods=max(1, window // 5))
          .mean()
          .reset_index(level=0, drop=True)
    )
    return df


# -----------------------------
# Cognitive Phases (derived)
# -----------------------------
PHASE_ORDER = [
    "exploracao_caotica",
    "exploracao_guiada",
    "transicao",
    "consolidacao",
    "estabilidade",
]

PHASE_LABEL = {
    "exploracao_caotica": "Exploracao Caotica",
    "exploracao_guiada": "Exploracao Guiada",
    "transicao": "Transicao",
    "consolidacao": "Consolidacao",
    "estabilidade": "Estabilidade",
}


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def classify_cognitive_phase_row(
    eps: float,
    mean_reward: float,
    slope: float,
    reward_std: float,
    *,
    eps_chaos: float = 0.80,
    eps_guided_low: float = 0.30,
    eps_consolidation: float = 0.10,
    std_stable: float = 0.15,
    slope_stable: float = 0.0,
    threshold: float = 0.0,
) -> str:
    """
    Classifica fase cognitiva a partir de sinais ja calculados.
    Regra e simples e interpretavel (CSV-first).
    """
    eps = _safe_float(eps)
    mr = _safe_float(mean_reward)
    slope = _safe_float(slope)
    std = _safe_float(reward_std)

    if eps >= eps_chaos:
        return "exploracao_caotica"

    if eps_guided_low <= eps < eps_chaos:
        return "exploracao_guiada"

    if mr >= threshold and eps < eps_guided_low:
        if std > std_stable:
            return "transicao"
        if eps < eps_consolidation:
            if std <= std_stable and slope >= slope_stable:
                return "estabilidade"
        return "consolidacao"

    return "consolidacao" if eps < eps_guided_low else "exploracao_guiada"


def add_cognitive_phases(
    df: pd.DataFrame,
    window: int = 50,
    *,
    threshold: float = 0.0,
    eps_chaos: float = 0.80,
    eps_guided_low: float = 0.30,
    eps_consolidation: float = 0.10,
    std_stable: float = 0.15,
    slope_stable: float = 0.0,
) -> pd.DataFrame:
    """
    Adiciona:
      - reward_ma, mean_reward_ma (se existirem)
      - reward_std_w (std janela)
      - slope_w (inclinacao janela do mean_reward)
      - cognitive_phase (label textual)
    """
    if df.empty:
        return df

    out = df.copy()

    if "episode" in out.columns:
        out["episode"] = pd.to_numeric(out["episode"], errors="coerce")
        out = out.dropna(subset=["episode"]).sort_values(["run_name", "episode"])
        out["episode"] = out["episode"].astype(int)

    def _per_run(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("episode").copy()

        if "reward" in g.columns:
            g["reward"] = pd.to_numeric(g["reward"], errors="coerce")
            g["reward_ma"] = (
                g["reward"]
                .rolling(window=window, min_periods=max(1, window // 5))
                .mean()
            )
            g["reward_std_w"] = (
                g["reward"]
                .rolling(window=window, min_periods=max(2, window // 5))
                .std(ddof=0)
            )
        else:
            g["reward_std_w"] = np.nan

        if "mean_reward" in g.columns:
            g["mean_reward"] = pd.to_numeric(g["mean_reward"], errors="coerce")
            g["mean_reward_ma"] = (
                g["mean_reward"]
                .rolling(window=window, min_periods=max(1, window // 5))
                .mean()
            )

        mr_col = None
        for c in ["mean_reward", "mean_reward_50", "mean_reward_ma", "reward_ma"]:
            if c in g.columns:
                mr_col = c
                break

        if mr_col is None:
            g["slope_w"] = np.nan
        else:
            y = pd.to_numeric(g[mr_col], errors="coerce").to_numpy()
            slopes = [np.nan] * len(g)
            ep = g["episode"].to_numpy()
            for i in range(len(g)):
                j0 = max(0, i - window + 1)
                ywin = y[j0:i + 1]
                epwin = ep[j0:i + 1]
                if len(ywin) >= max(5, window // 5) and np.isfinite(ywin).sum() >= 5:
                    mask = np.isfinite(ywin) & np.isfinite(epwin)
                    if mask.sum() >= 5:
                        slopes[i] = float(np.polyfit(epwin[mask], ywin[mask], 1)[0])
            g["slope_w"] = slopes

        if "mean_reward" not in g.columns:
            if "mean_reward_50" in g.columns:
                g["mean_reward"] = pd.to_numeric(g["mean_reward_50"], errors="coerce")
            elif "reward_ma" in g.columns:
                g["mean_reward"] = pd.to_numeric(g["reward_ma"], errors="coerce")
            elif "reward" in g.columns:
                g["mean_reward"] = pd.to_numeric(g["reward"], errors="coerce")
            else:
                g["mean_reward"] = np.nan

        if "epsilon" in g.columns:
            g["epsilon"] = pd.to_numeric(g["epsilon"], errors="coerce")
        else:
            g["epsilon"] = np.nan

        g["cognitive_phase"] = [
            classify_cognitive_phase_row(
                eps=g["epsilon"].iloc[i],
                mean_reward=g["mean_reward"].iloc[i],
                slope=g["slope_w"].iloc[i],
                reward_std=g["reward_std_w"].iloc[i],
                threshold=threshold,
                eps_chaos=eps_chaos,
                eps_guided_low=eps_guided_low,
                eps_consolidation=eps_consolidation,
                std_stable=std_stable,
                slope_stable=slope_stable,
            )
            for i in range(len(g))
        ]
        g["cognitive_phase_label"] = g["cognitive_phase"].map(PHASE_LABEL).fillna(
            g["cognitive_phase"]
        )
        return g

    out = out.groupby("run_name", group_keys=False).apply(_per_run)
    return out


def phase_segments(g: pd.DataFrame) -> list[dict]:
    """
    Retorna segmentos continuos de fase:
      [{"phase":..., "label":..., "start":..., "end":...}, ...]
    """
    if g.empty or "cognitive_phase" not in g.columns or "episode" not in g.columns:
        return []

    g = g.sort_values("episode").copy()
    phases = g["cognitive_phase"].tolist()
    eps = g["episode"].tolist()

    segs = []
    cur = phases[0]
    start = eps[0]
    for i in range(1, len(g)):
        if phases[i] != cur:
            segs.append(
                {
                    "phase": cur,
                    "label": PHASE_LABEL.get(cur, cur),
                    "start": int(start),
                    "end": int(eps[i - 1]),
                }
            )
            cur = phases[i]
            start = eps[i]
    segs.append(
        {
            "phase": cur,
            "label": PHASE_LABEL.get(cur, cur),
            "start": int(start),
            "end": int(eps[-1]),
        }
    )
    return segs


def plot_reward_with_phases(
    g: pd.DataFrame,
    *,
    title: str = "",
    use_mean_reward: bool = True,
    alpha_bg: float = 0.10,
):
    """
    Plota reward + (mean_reward opcional) com faixas de fase ao fundo.
    """
    if g.empty:
        return None

    g = g.sort_values("episode").copy()
    fig = plt.figure(figsize=(10, 3.2))
    ax = fig.add_subplot(111)

    if "reward" in g.columns:
        ax.plot(
            g["episode"],
            pd.to_numeric(g["reward"], errors="coerce"),
            alpha=0.25,
            label="reward",
        )

    if use_mean_reward and "mean_reward" in g.columns:
        ax.plot(
            g["episode"],
            pd.to_numeric(g["mean_reward"], errors="coerce"),
            linewidth=2.0,
            label="mean_reward",
        )

    segs = phase_segments(g)
    shade = True
    for s in segs:
        if shade:
            ax.axvspan(s["start"], s["end"], alpha=alpha_bg)
        shade = not shade

    ax.set_title(title or "Reward + Fases Cognitivas")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    return fig


def cognitive_phase_summary(g: pd.DataFrame) -> dict:
    """
    Resumo: fase atual, episodio de entrada, fase dominante.
    """
    if g.empty or "cognitive_phase" not in g.columns or "episode" not in g.columns:
        return {}

    g = g.sort_values("episode").copy()
    segs = phase_segments(g)
    current = segs[-1] if segs else None

    dom = (
        g["cognitive_phase"]
        .value_counts()
        .rename_axis("phase")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    dominant = dom.iloc[0]["phase"] if len(dom) else None

    entered_at = int(current["start"]) if current else None
    cur_phase = current["phase"] if current else None

    return {
        "current": cur_phase,
        "current_label": PHASE_LABEL.get(cur_phase, cur_phase),
        "entered_at_episode": entered_at,
        "dominant": dominant,
        "dominant_label": PHASE_LABEL.get(dominant, dominant),
        "segments": segs,
    }


def _to_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["__".join([str(x) for x in col]) for col in df.columns.to_list()]
    return df


def _safe_line_chart(pivot: pd.DataFrame, height: int = 240) -> None:
    pivot = _flatten_columns(pivot)
    pivot = pivot.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if pivot.empty:
        st.info("Sem dados numericos suficientes para plotar.")
        return
    st.line_chart(pivot, height=height, use_container_width=True)


def plot_rl_curve_single_metric(
    df: pd.DataFrame,
    runs: list[str],
    metric: str,
    window: int,
) -> None:
    if df.empty or not runs:
        st.info("Sem dados ou nenhum run selecionado.")
        return

    df = df[df["run_name"].isin(runs)].copy()
    if df.empty:
        st.info("Sem dados para os runs selecionados.")
        return

    if "mean_reward" not in df.columns and "mean_reward_50" in df.columns:
        df["mean_reward"] = df["mean_reward_50"]

    for c in [
        "episode",
        "reward",
        "mean_reward",
        "epsilon",
        "loss",
        "episode_length",
        "dream_loss",
        "dream_novelty_rate",
        "dream_novelty",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if metric in df.columns:
        df = add_rolling(df, metric, window=window)

    cols = []
    if metric in df.columns:
        cols.append(metric)
    if f"{metric}_ma" in df.columns:
        cols.append(f"{metric}_ma")

    if not cols:
        st.warning(f"Coluna '{metric}' nao existe neste historico.")
        return

    chart_df = df[["episode", "run_name"] + cols].copy()
    pivot = chart_df.pivot_table(index="episode", columns="run_name", values=cols)
    _safe_line_chart(pivot)


def plot_rl_curves(df: pd.DataFrame, runs: list[str], window: int) -> None:
    if df.empty:
        st.info("Nenhum rl_history.csv encontrado em runs/*/rl_history.csv")
        return

    if not runs:
        st.info("Selecione pelo menos 1 run para ver curvas.")
        return

    df = df[df["run_name"].isin(runs)].copy()
    if df.empty:
        st.info("Sem dados para os runs selecionados.")
        return

    base_numeric = [
        "episode",
        "reward",
        "mean_reward",
        "mean_reward_50",
        "epsilon",
        "loss",
        "episode_length",
        "dream_loss",
        "dream_novelty",
        "dream_novelty_rate",
        "dream_mix_prob",
        "dream_sigma",
        "dream_steps",
    ]
    df = _to_numeric_cols(df, base_numeric)

    if "mean_reward" not in df.columns and "mean_reward_50" in df.columns:
        df["mean_reward"] = df["mean_reward_50"]

    for col in [
        "reward",
        "mean_reward",
        "epsilon",
        "loss",
        "episode_length",
        "dream_loss",
        "dream_novelty",
        "dream_novelty_rate",
    ]:
        if col in df.columns:
            df = add_rolling(df, col, window=window)

    if "episode" not in df.columns:
        st.warning("Coluna 'episode' nao encontrada em rl_history.csv.")
        return

    st.caption("Escolha a metrica para plotar (mais leve e estavel).")
    metric = st.selectbox(
        "Metrica",
        options=[
            "reward",
            "mean_reward",
            "epsilon",
            "loss",
            "episode_length",
            "dream_loss",
            "dream_novelty_rate",
        ],
        index=1,
        key="rl_metric_select",
    )

    cols = []
    if metric in df.columns:
        cols.append(metric)
    if f"{metric}_ma" in df.columns:
        cols.append(f"{metric}_ma")

    if not cols:
        st.warning(f"Coluna '{metric}' nao encontrada neste rl_history.csv.")
        return

    chart_df = df[["episode", "run_name"] + cols].copy()
    pivot = chart_df.pivot_table(index="episode", columns="run_name", values=cols)
    st.subheader(f"Curva: {metric}")
    try:
        _safe_line_chart(pivot)
    except Exception as e:
        st.error(f"Falha ao plotar {metric}: {e}")
        st.stop()

    st.subheader("Ultimos pontos (por run)")
    last_rows = (
        df.sort_values(["run_name", "episode"])
          .groupby("run_name")
          .tail(1)
          .sort_values("run_name")
    )
    st.dataframe(last_rows, use_container_width=True)


def cognitive_metrics(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    out_rows = []

    for run, g in df.groupby("run_name"):
        g = g.sort_values("episode").copy()
        if g.empty:
            continue

        tail = g.tail(window) if len(g) >= window else g
        last = g.iloc[-1]
        mean_reward_last = float(last["mean_reward"]) if "mean_reward" in g.columns else float(tail["reward"].mean())
        reward_std = float(tail["reward"].std(ddof=0)) if "reward" in tail.columns else np.nan

        if "mean_reward" in tail.columns and len(tail) >= 5:
            x = tail["episode"].to_numpy()
            y = tail["mean_reward"].to_numpy()
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = np.nan

        if "epsilon" in tail.columns and "mean_reward" in tail.columns and len(tail) >= 2:
            eps0, eps1 = float(tail["epsilon"].iloc[0]), float(tail["epsilon"].iloc[-1])
            mr0, mr1 = float(tail["mean_reward"].iloc[0]), float(tail["mean_reward"].iloc[-1])
            de = (eps0 - eps1)
            dm = (mr1 - mr0)
            expl_eff = float(dm / de) if abs(de) > 1e-9 else np.nan
        else:
            expl_eff = np.nan

        if "episode_length" in tail.columns:
            len_mean = float(tail["episode_length"].mean())
            len_last = float(last["episode_length"])
        else:
            len_mean = np.nan
            len_last = np.nan

        out_rows.append({
            "run_name": run,
            "episodes": int(last["episode"]) + 1 if "episode" in g.columns else len(g),
            "mean_reward_last": mean_reward_last,
            "reward_std_last_window": reward_std,
            "learning_velocity_slope": slope,
            "exploration_efficiency": expl_eff,
            "episode_length_mean_last_window": len_mean,
            "episode_length_last": len_last,
            "epsilon_last": float(last["epsilon"]) if "epsilon" in g.columns else np.nan,
            "timestamp_last": str(last["timestamp"]) if "timestamp" in g.columns else "",
        })

    return pd.DataFrame(out_rows).sort_values("mean_reward_last", ascending=False)


def try_read_dream_csv(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "dream_history.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _try_import_tb():
    try:
        from tensorboard.backend.event_processing import event_accumulator
        return event_accumulator
    except Exception:
        return None


def _find_tfevents(run_dir: str) -> list[str]:
    pats = [
        os.path.join(run_dir, "**", "events.out.tfevents.*"),
        os.path.join(run_dir, "events.out.tfevents.*"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


@st.cache_data(show_spinner=False)
def read_dream_from_tensorboard(run_dir: str, tags: list[str] | None = None) -> pd.DataFrame | None:
    event_accumulator = _try_import_tb()
    if event_accumulator is None:
        return None

    files = _find_tfevents(run_dir)
    if not files:
        return None

    if tags is None:
        tags = [
            "dream/loss",
            "dream/novelty_rate",
            "dream/mix_prob",
            "dream/sigma",
            "dream/steps",
        ]

    rows = []
    for f in files:
        try:
            ea = event_accumulator.EventAccumulator(
                f,
                size_guidance={event_accumulator.SCALARS: 0},
            )
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            wanted = [t for t in tags if t in available]
            if not wanted:
                continue

            for t in wanted:
                for e in ea.Scalars(t):
                    rows.append({"tag": t, "step": int(e.step), "value": float(e.value)})
        except Exception:
            continue

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values(["tag", "step"])


@st.cache_data(show_spinner=False)
def summarize_dream_metrics(run_dir: str, window: int = 50) -> dict | None:
    df_csv = try_read_dream_csv(run_dir)
    if df_csv is not None and not df_csv.empty:
        df = df_csv.copy()

        if "night" not in df.columns:
            for c in ["nights", "dream_night", "dream/night"]:
                if c in df.columns:
                    df = df.rename(columns={c: "night"})
                    break

        last = df.iloc[-1]
        tail = df.tail(window)

        def _safe(col, default=np.nan):
            return float(last[col]) if col in df.columns else float(default)

        return {
            "source": "csv",
            "nights_detected": int(df["night"].nunique()) if "night" in df.columns else int(len(df)),
            "loss_last": _safe("loss"),
            "loss_mean_last_window": float(tail["loss"].mean()) if "loss" in tail.columns else float("nan"),
            "novelty_rate_last": _safe("novelty_rate"),
            "mix_prob_last": _safe("mix_prob"),
            "sigma_last": _safe("sigma"),
            "steps_last": _safe("steps"),
        }

    tb = read_dream_from_tensorboard(run_dir)
    if tb is None or tb.empty:
        return None

    def last_of(tag: str) -> float:
        g = tb[tb["tag"] == tag]
        return float(g["value"].iloc[-1]) if len(g) else float("nan")

    def mean_tail(tag: str, n: int) -> float:
        g = tb[tb["tag"] == tag].tail(n)
        return float(g["value"].mean()) if len(g) else float("nan")

    g_loss = tb[tb["tag"] == "dream/loss"]
    nights_detected = int(g_loss["step"].max()) if len(g_loss) else 0

    return {
        "source": "tensorboard",
        "nights_detected": int(nights_detected),
        "loss_last": last_of("dream/loss"),
        "loss_mean_last_window": mean_tail("dream/loss", window),
        "novelty_rate_last": last_of("dream/novelty_rate"),
        "mix_prob_last": last_of("dream/mix_prob"),
        "sigma_last": last_of("dream/sigma"),
        "steps_last": last_of("dream/steps"),
    }


def cognitive_metrics_single_run(g: pd.DataFrame, window: int = 50) -> dict:
    g = g.sort_values("episode").copy()
    tail = g.tail(window) if len(g) >= window else g
    last = g.iloc[-1]

    mean_reward_last = float(last["mean_reward"]) if "mean_reward" in g.columns else float(tail["reward"].mean())
    reward_std = float(tail["reward"].std(ddof=0)) if "reward" in tail.columns else float("nan")

    if "mean_reward" in tail.columns and len(tail) >= 5:
        x = tail["episode"].to_numpy()
        y = tail["mean_reward"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = float("nan")

    if "epsilon" in tail.columns and "mean_reward" in tail.columns and len(tail) >= 2:
        eps0, eps1 = float(tail["epsilon"].iloc[0]), float(tail["epsilon"].iloc[-1])
        mr0, mr1 = float(tail["mean_reward"].iloc[0]), float(tail["mean_reward"].iloc[-1])
        de = (eps0 - eps1)
        dm = (mr1 - mr0)
        expl_eff = float(dm / de) if abs(de) > 1e-9 else float("nan")
    else:
        expl_eff = float("nan")

    if "episode_length" in tail.columns:
        len_mean = float(tail["episode_length"].mean())
        len_last = float(last["episode_length"])
    else:
        len_mean = float("nan")
        len_last = float("nan")

    epsilon_last = float(last["epsilon"]) if "epsilon" in g.columns else float("nan")

    return {
        "window": int(window),
        "mean_reward_last": mean_reward_last,
        "reward_std_last_window": reward_std,
        "learning_velocity_slope": slope,
        "exploration_efficiency": expl_eff,
        "epsilon_last": epsilon_last,
        "episode_length_mean_last_window": len_mean,
        "episode_length_last": len_last,
    }


def detect_first_turning_point(g: pd.DataFrame, window: int = 50, threshold: float = 0.0):
    g = g.sort_values("episode").copy()
    if "mean_reward" not in g.columns or len(g) < max(window, 5):
        return None, None

    mr = g["mean_reward"].to_numpy()
    ep = g["episode"].to_numpy()

    for i in range(1, len(g)):
        if mr[i - 1] < threshold and mr[i] >= threshold:
            return int(ep[i]), f"mean_reward cruzou {threshold}"

    if len(g) >= window:
        for i in range(window, len(g) + 1):
            tail = g.iloc[i - window:i]
            x = tail["episode"].to_numpy()
            y = tail["mean_reward"].to_numpy()
            slope = float(np.polyfit(x, y, 1)[0])
            if slope > 0:
                return int(tail["episode"].iloc[-1]), "learning velocity ficou positiva"

    return None, None


def logan_reflection_text(payload: dict, run_name: str) -> str:
    reflection = payload["reflection"]
    return f"**{reflection['title']} (run {run_name})**\n\n{reflection['text']}"


def make_logan_reflection_json(
    g: pd.DataFrame,
    run_name: str,
    run_dir: str,
    device: str = "unknown",
    window: int = 50,
    threshold: float = 0.0,
    include_dream: bool = True,
) -> dict:
    g = g.sort_values("episode").copy()
    episodes = int(g["episode"].max()) + 1 if "episode" in g.columns else len(g)

    ts_first = str(g["timestamp"].iloc[0]) if "timestamp" in g.columns and len(g) else ""
    ts_last = str(g["timestamp"].iloc[-1]) if "timestamp" in g.columns and len(g) else ""

    signals = cognitive_metrics_single_run(g, window=window)
    if "run_name" not in g.columns:
        g = g.copy()
        g["run_name"] = run_name
    g_ph = add_cognitive_phases(g, window=window, threshold=threshold)
    phase_info = cognitive_phase_summary(g_ph) if not g_ph.empty else {}
    signals["cognitive_phase"] = {
        "current": phase_info.get("current"),
        "current_label": phase_info.get("current_label"),
        "entered_at_episode": phase_info.get("entered_at_episode"),
        "dominant": phase_info.get("dominant"),
        "dominant_label": phase_info.get("dominant_label"),
    }
    if include_dream:
        dream_summary = summarize_dream_metrics(run_dir, window=window)
        if dream_summary is not None:
            signals["dream"] = dream_summary
    turn_ep, reason = detect_first_turning_point(g, window=window, threshold=threshold)

    if turn_ep is None:
        summary = "Ainda nao detectei uma virada clara; estou consolidando padrao e reduzindo ruido."
        text = (
            f"Eu ainda nao detectei uma virada clara. "
            f"Mesmo assim, ha um padrao emergindo: mean_reward_final={signals['mean_reward_last']:.4f}, "
            f"epsilon_final={signals['epsilon_last']:.3f}. "
            f"Meu foco agora e estabilizar decisoes boas e reduzir ruido."
        )
        tags = ["pre-limiar", "exploracao", "estabilizacao"]
        milestone = None
    else:
        summary = "Atravessei meu primeiro limiar quando a performance deixou de ser acaso e virou padrao."
        text = (
            f"Eu atravessei meu primeiro limiar no episodio {turn_ep} ({reason}). "
            f"Antes disso, eu explorava em excesso. Depois, comecei a repetir o que funciona. "
            f"Agora: mean_reward_final={signals['mean_reward_last']:.4f}, "
            f"epsilon_final={signals['epsilon_last']:.3f}. "
            f"Meu proximo passo e transformar acerto em habito - sem perder curiosidade."
        )
        if "dream" in signals:
            tags = ["limiar", "aprendizado", "sonhos", "estabilizacao", "curiosidade"]
        else:
            tags = ["limiar", "aprendizado", "estabilizacao", "curiosidade"]
        milestone = {"episode": int(turn_ep), "reason": str(reason), "threshold": float(threshold)}

    cp = signals.get("cognitive_phase", {})
    if cp.get("current_label"):
        text += f" Fase atual: {cp['current_label']} (desde ep {cp.get('entered_at_episode')})."

    return {
        "schema_version": "logan_reflection_v2",
        "run": {
            "run_name": run_name,
            "episodes": episodes,
            "device": device,
            "timestamp_first": ts_first,
            "timestamp_last": ts_last,
        },
        "signals": signals,
        "milestones": {"first_turning_point": milestone},
        "reflection": {
            "title": "Primeira Reflexao Logan",
            "summary": summary,
            "text": text,
            "tags": tags,
        },
    }


def save_logan_reflection_json(payload: dict, run_dir: str, filename: str = "logan_reflection.json") -> str:
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("LAD Dashboard")
mode = st.sidebar.radio("Modo", ["Supervisionado (ML)", "Reinforcement Learning (Logan)"])

st.sidebar.divider()
st.sidebar.caption("Arquivos esperados em `reports/`.")
paths = {
    "eval_results": "reports/eval_results.csv",
    "leaderboard": "reports/leaderboard.csv",
    "rl_results": "reports/rl_results.csv",
    "leaderboard_rl": "reports/leaderboard_rl.csv",
}

# -----------------------------
# Main
# -----------------------------
st.title("LAD - Laboratory Control Panel")
st.caption("CSV-first. Reprodutivel. Comparavel. Visual.")

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("eval_results.csv", "OK" if os.path.exists(paths["eval_results"]) else "-")
with colB:
    st.metric("leaderboard.csv", "OK" if os.path.exists(paths["leaderboard"]) else "-")
with colC:
    st.metric("rl_results.csv", "OK" if os.path.exists(paths["rl_results"]) else "-")
with colD:
    st.metric("leaderboard_rl.csv", "OK" if os.path.exists(paths["leaderboard_rl"]) else "-")

st.divider()

# -----------------------------
# Supervised (ML)
# -----------------------------
if mode == "Supervisionado (ML)":
    section_title("Supervisionado - Resultados", "Le `reports/eval_results.csv` e `reports/leaderboard.csv`")

    df_eval = load_csv(paths["eval_results"])
    df_lb = load_csv(paths["leaderboard"])

    if df_eval is None:
        st.warning("Nao encontrei `reports/eval_results.csv`. Rode: `python -m src.cli eval --save-csv`")
        st.stop()

    # Normaliza colunas se existirem
    for col in ["val_loss", "val_acc"]:
        if col in df_eval.columns:
            df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

    # Filtros
    st.sidebar.subheader("Filtros (ML)")
    run_names = sorted(df_eval["run_name"].dropna().unique().tolist()) if "run_name" in df_eval.columns else []
    selected_runs = st.sidebar.multiselect("Runs", run_names, default=run_names[:5] if run_names else [])
    metric = st.sidebar.selectbox("Metrica principal", ["val_loss", "val_acc"])
    top_n = st.sidebar.slider("Top N", 3, 50, 10)

    df_show = df_eval.copy()
    if selected_runs and "run_name" in df_show.columns:
        df_show = df_show[df_show["run_name"].isin(selected_runs)]

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Tabela (eval_results)")
        st.dataframe(df_show.sort_values(metric, ascending=(metric == "val_loss")), use_container_width=True)

    with c2:
        st.markdown("### Top runs (agregado)")
        # Agrega por run_name: melhor val_loss e melhor val_acc
        agg = df_show.groupby("run_name", as_index=False).agg(
            best_val_loss=("val_loss", "min"),
            best_val_acc=("val_acc", "max"),
            n_evals=("run_name", "count"),
        )
        if metric == "val_loss":
            agg = agg.sort_values("best_val_loss", ascending=True).head(top_n)
        else:
            agg = agg.sort_values("best_val_acc", ascending=False).head(top_n)
        st.dataframe(agg, use_container_width=True)

    st.divider()

    st.markdown("### Grafico (Top runs)")
    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.add_subplot(111)

    if metric == "val_loss":
        ax.bar(agg["run_name"], agg["best_val_loss"])
        ax.set_ylabel("best val_loss (melhor para baixo)")
        ax.set_title("Top runs por val_loss")
    else:
        ax.bar(agg["run_name"], agg["best_val_acc"])
        ax.set_ylabel("best val_acc (melhor para cima)")
        ax.set_title("Top runs por val_acc")

    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig, clear_figure=True)

    if df_lb is not None:
        st.divider()
        st.markdown("### Leaderboard (arquivo)")
        st.dataframe(df_lb, use_container_width=True)

# -----------------------------
# RL (Logan)
# -----------------------------
else:
    section_title("RL - Logan Agent (PRO)", "Curvas | Reflexoes | Sonhos | Diagnostico")

    df_hist = discover_rl_history_cached("runs")
    available_runs = sorted(df_hist["run_name"].unique().tolist()) if not df_hist.empty else []

    if not available_runs:
        st.info(
            "Sem runs RL detectados ainda. Rode: `python -m src.cli rl-train` "
            "para gerar runs/<run>/rl_history.csv"
        )
        st.stop()

    st.sidebar.subheader("Selecao (PRO)")
    default_runs = available_runs[-1:]
    selected = st.sidebar.multiselect("Runs (RL)", options=available_runs, default=default_runs)
    smooth = st.sidebar.slider("Smoothing (media movel)", 1, 200, 50, 1)
    st.sidebar.caption("Dica: 25-50 = leitura limpa; 1 = sinal cru.")

    df_hist = add_cognitive_phases(df_hist, window=int(smooth), threshold=0.0)

    tabs = st.tabs(["Curvas", "Reflexoes", "Sonhos", "Diagnostico"])

    with tabs[0]:
        with safe_block("Curvas RL"):
            st.subheader("Curvas RL (CSV-first)")
            metric = st.selectbox(
                "Metrica",
                ["mean_reward", "reward", "epsilon", "loss", "episode_length", "dream_loss", "dream_novelty_rate"],
                index=0,
                key="pro_metric",
            )
            plot_rl_curve_single_metric(df_hist, runs=selected, metric=metric, window=smooth)

            st.subheader("Fases Cognitivas (timeline)")
            if selected:
                rn_focus = st.selectbox(
                    "Run foco (fases)",
                    options=selected,
                    index=len(selected) - 1,
                    key="phase_focus",
                )
                g_focus = df_hist[df_hist["run_name"] == rn_focus].copy()

                if not g_focus.empty:
                    summ = cognitive_phase_summary(g_focus)

                    cX, cY, cZ = st.columns(3)
                    with cX:
                        st.metric("Fase atual", summ.get("current_label", "-"))
                    with cY:
                        st.metric(
                            "Entrou no episodio", str(summ.get("entered_at_episode", "-"))
                        )
                    with cZ:
                        st.metric("Fase dominante", summ.get("dominant_label", "-"))

                    fig = plot_reward_with_phases(
                        g_focus,
                        title=f"{rn_focus} - Reward + Fases Cognitivas",
                        use_mean_reward=True,
                    )
                    st.pyplot(fig, clear_figure=True)

                    with st.expander("Segmentos detectados", expanded=False):
                        st.dataframe(
                            pd.DataFrame(summ.get("segments", [])),
                            use_container_width=True,
                        )

            st.divider()
            st.subheader("Ultimo ponto por run")
            last_rows = (
                df_hist[df_hist["run_name"].isin(selected)]
                .sort_values(["run_name", "episode"])
                .groupby("run_name")
                .tail(1)
                .sort_values("run_name")
            )
            st.dataframe(last_rows, use_container_width=True)

    with tabs[1]:
        with safe_block("Reflexoes Logan"):
            st.subheader("Primeira Reflexao Logan (auto)")
            ref_window = st.slider(
                "Janela para detectar limiar (W)", 20, 300, 50, 5, key="pro_ref_window"
            )
            threshold = st.number_input(
                "Threshold (mean_reward)", value=0.0, step=0.1, key="pro_threshold"
            )
            auto_save = st.checkbox(
                "Salvar JSON automaticamente no diretorio do run", value=True, key="pro_autosave"
            )

            for rn in selected:
                g = df_hist[df_hist["run_name"] == rn].copy()
                if g.empty:
                    continue
                run_dir = os.path.join("runs", rn)

                payload = make_logan_reflection_json(
                    g=g,
                    run_name=rn,
                    run_dir=run_dir,
                    device="unknown",
                    window=int(ref_window),
                    threshold=float(threshold),
                )

                st.markdown(logan_reflection_text(payload, rn))
                st.json(payload)

                if auto_save:
                    out_path = save_logan_reflection_json(payload, run_dir=run_dir)
                    st.caption(f"Salvo em: {out_path}")

            st.divider()
            st.subheader("Metricas cognitivas (comparativo)")
            metrics_window = st.slider(
                "Janela cognitiva (W)", 10, 300, 50, 5, key="pro_cog_window"
            )
            m = cognitive_metrics(
                df_hist[df_hist["run_name"].isin(selected)], window=int(metrics_window)
            )
            st.dataframe(m, use_container_width=True)

    with tabs[2]:
        with safe_block("Sonhos"):
            st.subheader("Sonhos (Dream) - resumo por run")
            for rn in selected:
                run_dir = os.path.join("runs", rn)
                summ = summarize_dream_metrics(run_dir, window=50)
                st.markdown(f"### {rn}")
                if summ is None:
                    st.info("Sem metricas de sonhos detectadas (CSV ou TensorBoard).")
                else:
                    st.json(summ)

            st.divider()
            st.subheader("Tabela de sonhos agregada (opcional)")
            dreams_path = "reports/rl_dreams.csv"
            if os.path.exists(dreams_path):
                df_d = pd.read_csv(dreams_path)
                if selected and "run_name" in df_d.columns:
                    df_d = df_d[df_d["run_name"].isin(selected)]
                st.dataframe(df_d.tail(200), use_container_width=True)
            else:
                st.info("`reports/rl_dreams.csv` ainda nao existe.")

    with tabs[3]:
        st.subheader("Diagnostico PRO")
        st.write("Ultimo erro capturado (se houver):")

        if "last_error" in st.session_state and st.session_state["last_error"]:
            e = st.session_state["last_error"]
            st.error(f"Contexto: {e.get('context','')}")
            st.code(e.get("traceback", ""))
            if st.button("Limpar erro", key="clear_last_error"):
                st.session_state["last_error"] = {}
                st.success("Erro limpo.")
        else:
            st.success("Nenhum erro registrado.")

        st.divider()
        st.subheader("Sanity checks")
        st.write("Runs detectados:", len(available_runs))
        st.write("Runs selecionados:", selected)

        for rn in selected:
            run_dir = Path("runs") / rn
            st.markdown(f"### {rn}")
            st.write("run_dir:", str(run_dir))
            st.write(
                "rl_history.csv:",
                "OK" if (run_dir / "rl_history.csv").exists() else "MISSING",
            )
            st.write(
                "logan_reflection.json:",
                "OK" if (run_dir / "logan_reflection.json").exists() else "MISSING",
            )
            ev = list(run_dir.glob("**/events.out.tfevents.*"))
            st.write("tfevents:", len(ev))

st.divider()
st.caption("LAD - CSV-first - runs/ + reports/ - Streamlit dashboard")
