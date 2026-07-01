"""
analyze_perf.py - Analises e graficos a partir de results_summary.csv (+ raw).

Gera, em results/analysis/<subpasta>/:
  complexity/      complexidade empirica log-log (tempo e instrucoes vs N) + expoente
  scalability/     heatmap de eficiencia, speedup vs threads + lei de Amdahl (fracao serial f)
  microarch/       scatter IPC x branch/cache-miss colorido por familia; heatmap normalizado
  memory/          pressao de L1/LLC/dTLB por algoritmo (N grande)
  distributions/   boxplots das 100 execucoes (tempo, cache-miss)            [precisa do raw]
  correlation/     matriz de correlacao contadores x tempo                   [usa raw se houver]
  reproducibility/ coeficiente de variacao por algoritmo (instr ~0 valida o metodo) [precisa do raw]

Uso:  python3 analyze_perf.py
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.autolayout'] = True

warnings.filterwarnings("ignore")

# ============================= Configuracao =============================
SUMMARY = "results/results_summary.csv"
RAW     = "results/results_raw.csv"
OUTDIR  = "results/analysis"
FIGFMT  = "pdf"          # formato das figuras
BASE_MODE, BASE_THREADS = "serial", 1   # baseline = "impressao digital" do algoritmo

# Mapeamento algoritmo -> familia (por palavra-chave no nome do binario)
FAMILY_KEYS = [
    ("Insertion",     ["insertion", "shell"]),
    ("Exchange",        ["comb", "quick", "bubble", "cocktail", "gnome"]),
    ("Selection",      ["heap", "smooth", "selection"]),
    ("Merge",        ["bottomup", "merge", "timsort"]),
    ("Distribution", ["count", "radix", "bucket", "pigeon"]),
    ("Network",         ["bitonic", "oddeven", "odd_even", "odd-even", "batcher"]),
]
FAMILY_COLOR = {
    "Insertion": "#1f77b4", "Exchange": "#ff7f0e", "Selection": "#2ca02c",
    "Merge": "#d62728", "Distribution": "#9467bd", "Network": "#8c564b",
    "Other": "#7f7f7f",
}

def family_of(algo):
    a = str(algo).lower()
    for fam, keys in FAMILY_KEYS:
        if any(k in a for k in keys):
            return fam
    return "Other"

def input_num(s):
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0

def ensure(sub):
    d = os.path.join(OUTDIR, sub)
    os.makedirs(d, exist_ok=True)
    return d

def save(fig, sub, name):
    d = ensure(sub)
    p = os.path.join(d, f"{name}.{FIGFMT}")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p

def annotate_heatmap(ax, M, fmt="{:.2f}", thr=None):
    vmax = np.nanmax(M) if np.isfinite(np.nanmax(M)) else 1.0
    vmin = np.nanmin(M) if np.isfinite(np.nanmin(M)) else 0.0
    mid = (vmax + vmin) / 2 if thr is None else thr
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=7, color="white" if v > mid else "black")

# ============================= Carga =============================
if not os.path.exists(SUMMARY):
    raise SystemExit(f"[ERRO] {SUMMARY} nao encontrado. Rode o parser primeiro.")
S = pd.read_csv(SUMMARY)
S["family"] = S["algo"].map(family_of)
S["Nnum"]   = S["input"].map(input_num)

R = None
if os.path.exists(RAW):
    R = pd.read_csv(RAW)
    R["family"] = R["algo"].map(family_of)
    R["Nnum"]   = R["input"].map(input_num)
else:
    print(f"[AVISO] {RAW} ausente: graficos de distribuicao/CV/correlacao(raw) serao pulados.")

algos  = sorted(S["algo"].unique())
sizes  = sorted(S["Nnum"].unique())
Nbig   = max(sizes)
threads_all = sorted(S["threads"].unique())

def base_slice(df):
    return df[(df["mode"] == BASE_MODE) & (df["threads"] == BASE_THREADS)]

def color_for(algo):
    return FAMILY_COLOR[family_of(algo)]

# ============================= 1. Complexidade =============================
def complexity():
    base = base_slice(S)
    for metric, ylabel, fname in [
        ("time_sec_mean",     "Time (s)",      "complexity_time"),
        ("instructions_mean", "Instructions",     "complexity_instructions"),
    ]:
        if metric not in base.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for algo in algos:
            sub = base[base["algo"] == algo].sort_values("Nnum")
            x = sub["Nnum"].values.astype(float)
            y = sub[metric].values.astype(float)
            ok = (x > 0) & (y > 0)
            x, y = x[ok], y[ok]
            if len(x) < 2:
                continue
            slope = np.polyfit(np.log(x), np.log(y), 1)[0]
            ax.plot(x, y, marker="o", color=color_for(algo),
                    label=f"{algo} (k≈{slope:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N (input size)"); ax.set_ylabel(ylabel)
        ax.set_title(f"Empirical complexity — {ylabel} vs $N$ (log-log, {BASE_MODE} t{BASE_THREADS})")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        save(fig, "complexity", fname)

# ============================= 2. Escalabilidade =============================
def efficiency_value(row):
    if "efficiency" in row and pd.notna(row["efficiency"]):
        return row["efficiency"]
    if "speedup" in row and pd.notna(row["speedup"]) and row["threads"]:
        return row["speedup"] / row["threads"]
    return np.nan

def scalability_heatmap():
    par = S[S["mode"] == "parallel"]
    if par.empty:
        print("[AVISO] sem dados paralelos: heatmap de eficiencia pulado.")
        return
    thr = sorted(par["threads"].unique())
    M = np.full((len(algos), len(thr)), np.nan)
    for i, algo in enumerate(algos):
        for j, t in enumerate(thr):
            rows = par[(par["algo"] == algo) & (par["threads"] == t)]
            vals = [efficiency_value(r) for _, r in rows.iterrows()]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                M[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(thr), 0.45 * len(algos) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"{t}t" for t in thr])
    ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos)
    ax.set_title("Parallel efficiency (speedup/threads), mean over $N$")
    annotate_heatmap(ax, M, "{:.2f}", thr=0.5)
    fig.colorbar(im, ax=ax, label="efficiency")
    save(fig, "scalability", "efficiency_heatmap")

def amdahl_fit(threads, speedups):
    """Estima fracao serial f minimizando SSE de S(p)=1/(f+(1-f)/p) por busca em grade."""
    t = np.array(threads, float); s = np.array(speedups, float)
    ok = np.isfinite(t) & np.isfinite(s) & (t > 0)
    t, s = t[ok], s[ok]
    if len(t) < 2:
        return np.nan
    best_f, best_sse = np.nan, np.inf
    for f in np.linspace(0.0, 1.0, 1001):
        pred = 1.0 / (f + (1.0 - f) / t)
        sse = np.sum((s - pred) ** 2)
        if sse < best_sse:
            best_sse, best_f = sse, f
    return best_f

def speedup_and_amdahl():
    # speedup vs threads no N grande, todos os algos + linha ideal
    sub = S[(S["Nnum"] == Nbig)]
    if "speedup" not in sub.columns:
        print("[AVISO] coluna speedup ausente: secao Amdahl pulada.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    fvals = {}
    for algo in algos:
        a = sub[sub["algo"] == algo]
        pts = []
        for t in threads_all:
            if t == 1:
                pts.append((1, 1.0)); continue
            r = a[(a["mode"] == "parallel") & (a["threads"] == t)]
            if len(r):
                sp = r.iloc[0].get("speedup", np.nan)
                if pd.notna(sp):
                    pts.append((t, sp))
        if len(pts) < 2:
            continue
        pts.sort()
        tt = [p[0] for p in pts]; ss = [p[1] for p in pts]
        ax.plot(tt, ss, marker="o", color=color_for(algo), label=algo)
        fvals[algo] = amdahl_fit(tt, ss)
    lim = max(threads_all)
    ax.plot([1, lim], [1, lim], "k--", alpha=0.5, label="ideal (y=x)")
    ax.set_xlabel("threads"); ax.set_ylabel("speedup")
    ax.set_title(f"Speedup vs threads ($N={Nbig}$) with ideal line")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save(fig, "scalability", "speedup_vs_threads")

    # barra da fracao serial estimada (Amdahl) por algoritmo
    if fvals:
        items = sorted(fvals.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=9)))
        names = [k for k, _ in items]; fs = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.5))
        cols = [color_for(a) for a in names]
        ax.barh(range(len(names)), fs, color=cols)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("estimated serial fraction f (Amdahl)")
        ax.set_title(f"Serial fraction per algorithm ($N={Nbig}$) — lower $f$ scales better")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        for i, v in enumerate(fs):
            if pd.notna(v):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        save(fig, "scalability", "amdahl_serial_fraction")

# ============================= 3. Microarquitetura =============================
def microarch_scatter():
    base = base_slice(S[S["Nnum"] == Nbig])
    if base.empty:
        base = base_slice(S)
    for xcol, xlabel, fname in [
        ("branch_miss_rate_mean", "Branch miss rate", "ipc_vs_branchmiss"),
        ("cache_miss_rate_mean",  "Cache miss rate",  "ipc_vs_cachemiss"),
    ]:
        if xcol not in base.columns or "ipc_mean" not in base.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        seen = set()
        for _, r in base.iterrows():
            fam = family_of(r["algo"])
            ax.scatter(r[xcol], r["ipc_mean"], s=90, color=FAMILY_COLOR[fam],
                       label=fam if fam not in seen else None,
                       edgecolor="black", linewidth=0.5, zorder=3)
            seen.add(fam)
            ax.annotate(r["algo"], (r[xcol], r["ipc_mean"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xlabel); ax.set_ylabel("IPC")
        ax.set_title(f"Microarchitecture fingerprint ($N={Nbig}$, {BASE_MODE})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="family", fontsize=8)
        save(fig, "microarch", fname)

def microarch_heatmap():
    base = base_slice(S[S["Nnum"] == Nbig])
    if base.empty:
        base = base_slice(S)
    cols = [c for c in ["ipc_mean", "cpi_mean", "branch_miss_rate_mean",
                        "cache_miss_rate_mean", "L1_dcache_load_misses_mean",
                        "LLC_load_misses_mean", "dTLB_load_misses_mean",
                        "iTLB_load_misses_mean"] if c in base.columns]
    if not cols:
        return
    base = base.copy()
    base["family"] = base["algo"].map(family_of)
    base = base.sort_values(["family", "algo"])
    names = base["algo"].tolist()
    M = base[cols].values.astype(float)
    # normalizacao min-max per column (contador)
    Mn = np.zeros_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        Mn[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.0
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 2, 0.45 * len(names) + 1.5))
    im = ax.imshow(Mn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_mean", "") for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"Normalized counter portrait ($N={Nbig}$) — sorted by family")
    fig.colorbar(im, ax=ax, label="min-max per column")
    save(fig, "microarch", "counter_fingerprint")

# ============================= 4. Memoria =============================
def memory_pressure():
    base = base_slice(S[S["Nnum"] == Nbig])
    cols = [c for c in ["L1_dcache_load_misses_mean", "LLC_load_misses_mean",
                        "dTLB_load_misses_mean", "iTLB_load_misses_mean"] if c in base.columns]
    if base.empty or not cols:
        return
    names = base["algo"].tolist()
    x = np.arange(len(names)); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 6))
    for k, c in enumerate(cols):
        ax.bar(x + k * w, base[c].values + 1, width=w, label=c.replace("_mean", ""))
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("misses (log, +1)")
    ax.set_title(f"Memory hierarchy pressure ($N={Nbig}$, {BASE_MODE})")
    ax.legend(fontsize=8); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save(fig, "memory", "memory_hierarchy")

# ============================= 5. Distribuicoes (raw) =============================
def distributions():
    if R is None:
        return
    sub = R[(R["mode"] == BASE_MODE) & (R["threads"] == BASE_THREADS) & (R["Nnum"] == Nbig)]
    if sub.empty:
        print("[AVISO] raw sem casos no baseline/N grande: distribuicoes puladas.")
        return
    for col, ylabel, fname in [
        ("time_sec",        "Time (s)",          "dist_time"),
        ("cache_miss_rate", "Cache miss rate", "dist_cachemiss"),
    ]:
        if col not in sub.columns:
            continue
        data, names = [], []
        for algo in algos:
            v = sub[sub["algo"] == algo][col].dropna().values
            if len(v):
                data.append(v); names.append(algo)
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 6))
        bp = ax.boxplot(data, labels=names, showfliers=True, patch_artist=True)
        for patch, algo in zip(bp["boxes"], names):
            patch.set_facecolor(color_for(algo)); patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Distribution over repetitions ($N={Nbig}$, {BASE_MODE})")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        save(fig, "distributions", fname)

# ============================= 6. Correlacao =============================
def correlation():
    src = R if R is not None else S
    cmap_cols = {
        "time_sec": "time_sec_mean", "ipc": "ipc_mean", "cpi": "cpi_mean",
        "branch_miss_rate": "branch_miss_rate_mean", "cache_miss_rate": "cache_miss_rate_mean",
        "L1_dcache_load_misses": "L1_dcache_load_misses_mean",
        "LLC_load_misses": "LLC_load_misses_mean",
        "dTLB_load_misses": "dTLB_load_misses_mean",
        "instructions": "instructions_mean", "cycles": "cycles_mean",
    }
    cols = []
    for raw_c, sum_c in cmap_cols.items():
        c = raw_c if (R is not None and raw_c in src.columns) else sum_c
        if c in src.columns:
            cols.append(c)
    if len(cols) < 3:
        return
    C = src[cols].corr()
    labels = [c.replace("_mean", "") for c in cols]
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 3, 0.7 * len(cols) + 2))
    im = ax.imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels, fontsize=8)
    annotate_heatmap(ax, C.values, "{:.2f}", thr=0.0)
    ax.set_title("Correlation between counters and time")
    fig.colorbar(im, ax=ax, label="Pearson r")
    save(fig, "correlation", "counter_correlation")

# ============================= 7. Reprodutibilidade (raw) =============================
def reproducibility():
    if R is None:
        return
    metrics = [m for m in ["instructions", "branches", "cycles", "time_sec"] if m in R.columns]
    if not metrics:
        return
    keys = ["algo", "input", "mode", "threads"]
    # CV por caso, depois media por algoritmo
    rows = []
    for algo in algos:
        cvs = {m: [] for m in metrics}
        for _, g in R[R["algo"] == algo].groupby(keys):
            for m in metrics:
                mu = g[m].mean()
                if mu:
                    cvs[m].append(g[m].std(ddof=1) / mu)
        rows.append({"algo": algo, **{m: (np.mean(cvs[m]) if cvs[m] else np.nan) for m in metrics}})
    D = pd.DataFrame(rows).set_index("algo")
    x = np.arange(len(D)); w = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(1.0 * len(D) + 2, 6))
    for k, m in enumerate(metrics):
        ax.bar(x + k * w, D[m].values + 1e-9, width=w, label=m)
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(D.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("coef. of variation (log)")
    ax.set_title(r"Reproducibility: CV per algorithm (instructions $\approx$ 0 validates the method)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save(fig, "reproducibility", "cv_by_algorithm")

# ============================= 8. Energia (RAPL) =============================
def has_energy():
    c = "energy_total_joules_mean"
    return c in S.columns and S[c].fillna(0).max() > 0

def select_row(df, algo, t):
    """Linha de (algo, t) escolhendo o modo: serial p/ 1 thread, parallel p/ 2+."""
    sub = df[(df["algo"] == algo) & (df["threads"] == t)]
    if len(sub) > 1:
        pref = "serial" if t == 1 else "parallel"
        s2 = sub[sub["mode"] == pref]
        sub = s2 if len(s2) else sub
    return sub.iloc[0] if len(sub) else None

def energy_scaling():
    if not has_energy():
        print("[AVISO] sem dados de energia (RAPL): graficos de energia pulados.")
        return
    sub = S[S["Nnum"] == Nbig]
    for metric, ylabel, fname, title in [
        ("energy_total_joules_mean", "Energy (J)", "energy_vs_threads",
         f"Total energy vs threads ($N={Nbig}$) — does not drop like time"),
        ("power_watts_mean", "Average power (W)", "power_vs_threads",
         f"Average power vs threads ($N={Nbig}$) — rises with active cores"),
    ]:
        if metric not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        drew = False
        for algo in algos:
            xs, ys = [], []
            for t in threads_all:
                r = select_row(sub, algo, t)
                if r is not None and pd.notna(r.get(metric)) and r.get(metric):
                    xs.append(t); ys.append(r[metric])
            if xs:
                ax.plot(xs, ys, marker="o", color=color_for(algo), label=algo)
                drew = True
        if not drew:
            plt.close(fig); continue
        ax.set_xlabel("threads"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        save(fig, "energy", fname)

def energy_time_pareto():
    if not has_energy():
        return
    sub = S[S["Nnum"] == Nbig]
    pts = []
    for algo in algos:
        for t in threads_all:
            r = select_row(sub, algo, t)
            if r is None:
                continue
            tm, en = r.get("time_sec_mean"), r.get("energy_total_joules_mean")
            if pd.notna(tm) and pd.notna(en) and tm > 0 and en > 0:
                pts.append((tm, en, algo, t))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for tm, en, algo, t in pts:
        fam = family_of(algo)
        ax.scatter(tm, en, s=60, color=FAMILY_COLOR[fam],
                   label=fam if fam not in seen else None,
                   edgecolor="black", linewidth=0.4, zorder=3)
        seen.add(fam)
        ax.annotate(f"{algo}·{t}t", (tm, en), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    # Fronteira de Pareto: minimizar tempo E energia (canto inferior esquerdo).
    P = sorted(pts, key=lambda p: (p[0], p[1]))
    front, best = [], float("inf")
    for tm, en, algo, t in P:
        if en < best - 1e-12:
            front.append((tm, en)); best = en
    if len(front) >= 2:
        ax.plot([p[0] for p in front], [p[1] for p in front],
                "k--", alpha=0.6, label="Pareto front", zorder=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Energy (J)")
    ax.set_title(f"Energy-time trade-off ($N={Nbig}$, all thread counts)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, "energy", "energy_time_pareto")

def edp_heatmap():
    if not has_energy():
        return
    sub = S[S["Nnum"] == Nbig]
    for metric, fname, title in [
        ("edp_mean",  "edp_heatmap",
         f"EDP (J·s) by algorithm x threads ($N={Nbig}$) — green = best per row"),
        ("ed2p_mean", "ed2p_heatmap",
         f"ED$^2$P (J·s$^2$) by algorithm x threads ($N={Nbig}$) — green = best per row"),
    ]:
        if metric not in sub.columns:
            continue
        M = np.full((len(algos), len(threads_all)), np.nan)
        for i, algo in enumerate(algos):
            for j, t in enumerate(threads_all):
                r = select_row(sub, algo, t)
                if r is not None and pd.notna(r.get(metric)) and r.get(metric):
                    M[i, j] = r[metric]
        if np.all(np.isnan(M)):
            continue
        # Normaliza por LINHA (algoritmo) para destacar a melhor config de cada um.
        Mn = np.full_like(M, np.nan)
        for i in range(M.shape[0]):
            row = M[i]; lo, hi = np.nanmin(row), np.nanmax(row)
            Mn[i] = (row - lo) / (hi - lo) if hi > lo else 0.0
        fig, ax = plt.subplots(figsize=(1.3 * len(threads_all) + 2, 0.45 * len(algos) + 1.5))
        im = ax.imshow(Mn, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(threads_all))); ax.set_xticklabels([f"{t}t" for t in threads_all])
        ax.set_yticks(range(len(algos))); ax.set_yticklabels(algos)
        ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="per-row min-max (green = lower)")
        save(fig, "energy", fname)

def has_ram_energy():
    c = "energy_ram_joules_mean"
    return c in S.columns and S[c].fillna(0).max() > 0

def energy_breakdown():
    if not has_energy():
        return
    if not has_ram_energy():
        print("[AVISO] dominio RAPL DRAM ausente: breakdown teria so package; pulado.")
        return
    sub = S[S["Nnum"] == Nbig]
    pkg, ram, names = [], [], []
    for algo in algos:
        r = select_row(sub, algo, 1)          # baseline serial t=1
        if r is None:
            continue
        p = r.get("energy_pkg_joules_mean", 0.0) or 0.0
        m = r.get("energy_ram_joules_mean", 0.0) or 0.0
        if (p + m) <= 0:
            continue
        pkg.append(p); ram.append(m); names.append(algo)
    if not names:
        return

    # (1) barra empilhada: package + DRAM em Joules absolutos
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 6))
    ax.bar(x, pkg, color="#4c78a8", label="Package (cores+cache+IMC)")
    ax.bar(x, ram, bottom=pkg, color="#f58518", label="DRAM")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Energy (J)")
    ax.set_title(f"Energy breakdown: package vs DRAM ($N={Nbig}$, serial)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    save(fig, "energy", "energy_breakdown_stacked")

    # (2) fracao da energia gasta em DRAM (ordenada) = "memory-bound energetico"
    frac = [m / (p + m) for p, m in zip(pkg, ram)]
    order = np.argsort(frac)                  # ascendente
    nn = [names[i] for i in order]; ff = [frac[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(nn) + 1.5))
    ax.barh(range(len(nn)), ff, color=[color_for(a) for a in nn])
    ax.set_yticks(range(len(nn))); ax.set_yticklabels(nn)
    ax.set_xlabel(r"DRAM energy share  ($E_{RAM} / E_{total}$)")
    ax.set_title(f"Share of energy spent in DRAM ($N={Nbig}$, serial) — higher = more memory-bound")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(ff):
        ax.text(v, i, f" {v*100:.0f}%", va="center", fontsize=8)
    save(fig, "energy", "energy_dram_share")

# ============================= Execucao =============================
STEPS = [
    ("complexidade",      complexity),
    ("eficiencia",        scalability_heatmap),
    ("speedup/amdahl",    speedup_and_amdahl),
    ("microarch scatter", microarch_scatter),
    ("microarch heatmap", microarch_heatmap),
    ("memoria",           memory_pressure),
    ("distribuicoes",     distributions),
    ("correlacao",        correlation),
    ("reprodutibilidade", reproducibility),
    ("energia (scaling)", energy_scaling),
    ("energia (pareto)",  energy_time_pareto),
    ("energia (edp)",     edp_heatmap),
    ("energia (breakdown)", energy_breakdown),
]

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[INFO] {len(algos)} algoritmos, {len(sizes)} tamanhos, threads={threads_all}")
    print(f"[INFO] familias: {sorted(set(family_of(a) for a in algos))}")
    done = 0
    for label, fn in STEPS:
        try:
            fn(); done += 1
        except Exception as e:
            print(f"[AVISO] '{label}' falhou: {type(e).__name__}: {e}")
    n_fig = sum(len(fs) for _, _, fs in os.walk(OUTDIR))
    print(f"[OK] {done}/{len(STEPS)} analises concluidas, {n_fig} figuras em {OUTDIR}/")
