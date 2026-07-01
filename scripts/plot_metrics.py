import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scipy e opcional: se ausente, usa aproximacao normal (1,96) para o IC 95%.
try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Padroniza todas as figuras num unico lugar.
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.autolayout'] = True

# =============================================================================
# Configuracao
# =============================================================================
RAW = "results/results_raw.csv"     # uma linha por execucao (saida do parser)
OUTDIR = "results/plots"

# Liga/desliga familias de saida (controle de quantos PDFs sao gerados).
PLOT_MEAN_CI = True    # media com barra de IC 95% (familias por-algo e por-thread)
PLOT_STD     = True    # desvio padrao como curva propria
PLOT_SPEEDUP = True    # speedup vs baseline

# Baseline do speedup:
#   "serial" -> T_serial(t=1) / T(t)    (speedup absoluto; cai p/ parallel t=1 se faltar serial)
#   "self"   -> T_parallel(t=1) / T(t)  (speedup relativo; cai p/ serial t=1 se faltar parallel)
BASELINE_MODE = "serial"

CONF_LEVEL = 0.95

# Escala do eixo X (input/N). "log" e recomendado quando N varia em ordens de
# grandeza (ex.: 1e5..1e7), pois espaca as potencias de dez uniformemente e evita
# rotulos sobrepostos. Use "linear" quando os N forem proximos/uniformes.
XSCALE = "log"

# Metricas plotadas: (coluna_no_csv, rotulo_do_eixo_y)
METRICS = [
    ("cpi",                   "CPI"),
    ("ipc",                   "IPC"),
    ("time_sec",              "Execution time (s)"),
    ("branch_miss_rate",      "Branch miss rate"),
    ("cache_miss_rate",       "Cache miss rate"),
    ("L1_dcache_load_misses", "L1 dcache load misses"),
    ("LLC_load_misses",       "LLC load misses"),
    ("dTLB_load_misses",      "dTLB load misses"),
    ("iTLB_load_misses",      "iTLB load misses"),
]

COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
MARKERS = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

GROUP_KEYS = ["algo", "input", "mode", "threads"]

# =============================================================================
# Carga e ordenacao
# =============================================================================
df = pd.read_csv(RAW)

def input_num(s):
    """Chave numerica para ordenar input1, input2, ... input10 corretamente."""
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0

inputs      = sorted(df["input"].unique(), key=input_num)
algos       = sorted(df["algo"].unique())
threads_all = sorted(df["threads"].unique())

os.makedirs(OUTDIR, exist_ok=True)
for sub in ["by_algo", "by_threads", "std_by_algo", "std_by_threads", "speedup"]:
    os.makedirs(os.path.join(OUTDIR, sub), exist_ok=True)

# =============================================================================
# Agregacao por caso: media, desvio padrao (ddof=1) e meia-largura do IC 95%,
# a partir das repeticoes (idealmente 100). Tudo calculado aqui, nao no parser.
# =============================================================================
def ci_halfwidth(std, n):
    if n is None or n < 2 or std is None or np.isnan(std):
        return 0.0
    sem = std / np.sqrt(n)
    if HAVE_SCIPY:
        tcrit = stats.t.ppf(0.5 + CONF_LEVEL / 2.0, n - 1)
    else:
        tcrit = 1.96  # aproximacao normal para n grande
    return tcrit * sem

agg_rows = []
metric_cols = [m[0] for m in METRICS] + ["time_sec"]
for keys, sub in df.groupby(GROUP_KEYS):
    algo, inp, mode, threads = keys
    row = {"algo": algo, "input": inp, "mode": mode,
           "threads": threads, "n": len(sub)}
    for col in set(metric_cols):
        vals = sub[col].dropna() if col in sub else pd.Series([], dtype=float)
        m = vals.mean() if len(vals) else np.nan
        s = vals.std(ddof=1) if len(vals) > 1 else 0.0
        row[f"{col}_mean"] = m
        row[f"{col}_std"]  = s
        row[f"{col}_ci"]   = ci_halfwidth(s, len(vals))
    agg_rows.append(row)
agg = pd.DataFrame(agg_rows)

# Aviso de reprodutibilidade: instructions deve ter desvio ~0 (validacao).
if "instructions" in df.columns:
    for keys, sub in df.groupby(GROUP_KEYS):
        s = sub["instructions"].std(ddof=1) if len(sub) > 1 else 0.0
        if s > 1.0:  # mais que ~1 instrucao de variacao e suspeito
            print(f"[AVISO] instructions varia ({s:.1f}) em "
                  f"{keys} -- algum controle de reprodutibilidade vazou.")

# =============================================================================
# Selecao de modo: para uma dada (algo, threads), escolhe a linha certa.
# Regra: 1 thread -> serial; 2+ threads -> parallel. Cai no que existir se o
# preferido faltar. ALTERE AQUI se seu desenho experimental for diferente.
# =============================================================================
def pick_row(algo, inp, threads):
    sel = agg[(agg.algo == algo) & (agg.input == inp) & (agg.threads == threads)]
    if len(sel) > 1:
        pref = "serial" if threads == 1 else "parallel"
        s2 = sel[sel["mode"] == pref]
        sel = s2 if len(s2) else sel
    return sel.iloc[0] if len(sel) else None

def series(algo, threads, metric):
    """Retorna (mean[], ci[], std[]) ao longo de inputs para (algo, threads)."""
    means, cis, stds = [], [], []
    for inp in inputs:
        r = pick_row(algo, inp, threads)
        if r is None or pd.isna(r.get(f"{metric}_mean", np.nan)):
            means.append(np.nan); cis.append(0.0); stds.append(0.0)
        else:
            means.append(r[f"{metric}_mean"])
            cis.append(r[f"{metric}_ci"])
            stds.append(r[f"{metric}_std"])
    return np.array(means), np.array(cis), np.array(stds)

# =============================================================================
# Rotina generica de plotagem (linhas vs input)
# =============================================================================
def draw(curves, ylabel, title, outpath, with_err):
    x = np.array([input_num(inp) for inp in inputs], dtype=float)  # N real, nao indice
    plt.figure(figsize=(10, 6))
    plotted = False
    for c in curves:
        if np.all(np.isnan(c["y"])):
            continue
        plotted = True
        if with_err:
            plt.errorbar(x, c["y"], yerr=c["err"], marker=c["marker"],
                         color=c["color"], label=c["label"], capsize=3,
                         linewidth=1.5, markersize=6)
        else:
            plt.plot(x, c["y"], marker=c["marker"], color=c["color"],
                     label=c["label"], linewidth=1.5, markersize=6)
    if not plotted:
        plt.close(); return
    plt.xscale(XSCALE)
    plt.xticks(x, inputs, rotation=45)
    plt.xlabel("Input size (N)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# =============================================================================
# FAMILIA 1: por algoritmo (linhas = threads)
# =============================================================================
def family_by_algo(use_std):
    for metric, ylabel in METRICS:
        for algo in algos:
            curves = []
            for i, t in enumerate(threads_all):
                mean, ci, std = series(algo, t, metric)
                curves.append({
                    "label": f"{t} thread" + ("s" if t > 1 else ""),
                    "y": std if use_std else mean,
                    "err": None if use_std else ci,
                    "color": COLORS[i % len(COLORS)],
                    "marker": MARKERS[i % len(MARKERS)],
                })
            if use_std:
                draw(curves, f"Standard deviation - {ylabel}",
                     f"Standard deviation of {ylabel} - {algo}",
                     f"{OUTDIR}/std_by_algo/{algo}_{metric}_std.pdf", with_err=False)
            else:
                draw(curves, ylabel,
                     f"{ylabel} - {algo} (mean +- {int(CONF_LEVEL*100)}% CI)",
                     f"{OUTDIR}/by_algo/{algo}_{metric}.pdf", with_err=True)

# =============================================================================
# FAMILIA 2: por numero de threads (linhas = algoritmos)
# =============================================================================
def family_by_threads(use_std):
    for metric, ylabel in METRICS:
        for t in threads_all:
            curves = []
            for i, algo in enumerate(algos):
                mean, ci, std = series(algo, t, metric)
                curves.append({
                    "label": algo,
                    "y": std if use_std else mean,
                    "err": None if use_std else ci,
                    "color": COLORS[i % len(COLORS)],
                    "marker": MARKERS[i % len(MARKERS)],
                })
            tl = f"{t} thread" + ("s" if t > 1 else "")
            if use_std:
                draw(curves, f"Standard deviation - {ylabel}",
                     f"Standard deviation of {ylabel} - {tl}",
                     f"{OUTDIR}/std_by_threads/t{t}_{metric}_std.pdf", with_err=False)
            else:
                draw(curves, ylabel,
                     f"{ylabel} - {tl} (mean +- {int(CONF_LEVEL*100)}% CI)",
                     f"{OUTDIR}/by_threads/t{t}_{metric}.pdf", with_err=True)

# =============================================================================
# SPEEDUP: baseline = serial t=1, por (algo, input). IC propagado pela razao.
# =============================================================================
def baseline_time(algo, inp):
    # Ordem de preferencia conforme BASELINE_MODE, com fallback para o outro.
    order = [("serial", 1), ("parallel", 1)] if BASELINE_MODE == "serial" \
            else [("parallel", 1), ("serial", 1)]
    for mode, t in order:
        sel = agg[(agg.algo == algo) & (agg.input == inp) &
                  (agg["mode"] == mode) & (agg.threads == t)]
        if len(sel):
            r = sel.iloc[0]
            return r["time_sec_mean"], r["time_sec_ci"]
    return np.nan, 0.0

def speedup_series(algo, threads):
    sp, err = [], []
    for inp in inputs:
        base_m, base_ci = baseline_time(algo, inp)
        r = pick_row(algo, inp, threads)
        par_m = r["time_sec_mean"] if r is not None else np.nan
        par_ci = r["time_sec_ci"] if r is not None else 0.0
        if not base_m or not par_m or np.isnan(base_m) or np.isnan(par_m):
            sp.append(np.nan); err.append(0.0); continue
        s = base_m / par_m
        # Propagacao de incerteza para a razao (erros relativos em quadratura).
        rel = np.sqrt((base_ci / base_m) ** 2 + (par_ci / par_m) ** 2)
        sp.append(s); err.append(s * rel)
    return np.array(sp), np.array(err)

def plot_speedup():
    multi = [t for t in threads_all if t > 1]
    if not multi:
        print("[AVISO] Speedup nao gerado: nao ha execucoes com mais de 1 "
              "thread nos dados (so existe threads=1). Rode o experimento "
              "paralelo para obter speedup.")
        return

    # Checa se ha baseline para pelo menos um caso.
    has_baseline = any(
        not np.isnan(baseline_time(algo, inp)[0])
        for algo in algos for inp in inputs
    )
    if not has_baseline:
        print(f"[AVISO] Speedup nao gerado: nenhum baseline encontrado "
              f"(BASELINE_MODE='{BASELINE_MODE}'). Verifique se existem linhas "
              f"com threads=1 no CSV. Modes disponiveis: "
              f"{sorted(agg['mode'].unique())}.")
        return

    saved = 0
    # Por algoritmo (linhas = threads > 1)
    for algo in algos:
        curves = []
        for i, t in enumerate(threads_all):
            if t == 1:
                continue
            sp, err = speedup_series(algo, t)
            curves.append({"label": f"{t} threads", "y": sp, "err": err,
                           "color": COLORS[i % len(COLORS)],
                           "marker": MARKERS[i % len(MARKERS)]})
        if any(not np.all(np.isnan(c["y"])) for c in curves):
            draw(curves, "Speedup (T_base / T_parallel)",
                 f"Speedup - {algo} (mean +- {int(CONF_LEVEL*100)}% CI)",
                 f"{OUTDIR}/speedup/{algo}_speedup.pdf", with_err=True)
            saved += 1

    # Comparacao entre algoritmos (linhas = algoritmos), por contagem de threads
    for t in threads_all:
        if t == 1:
            continue
        curves = []
        for i, algo in enumerate(algos):
            sp, err = speedup_series(algo, t)
            curves.append({"label": algo, "y": sp, "err": err,
                           "color": COLORS[i % len(COLORS)],
                           "marker": MARKERS[i % len(MARKERS)]})
        if any(not np.all(np.isnan(c["y"])) for c in curves):
            draw(curves, "Speedup (T_base / T_parallel)",
                 f"Speedup - {t} threads (mean +- {int(CONF_LEVEL*100)}% CI)",
                 f"{OUTDIR}/speedup/t{t}_speedup.pdf", with_err=True)
            saved += 1

    if saved == 0:
        print("[AVISO] Speedup: baseline existe, mas nenhum caso paralelo "
              "casou com o baseline (confira pares algo/input entre serial e "
              "parallel).")

# =============================================================================
# Execucao
# =============================================================================
if PLOT_MEAN_CI:
    family_by_algo(use_std=False)
    family_by_threads(use_std=False)
if PLOT_STD:
    family_by_algo(use_std=True)
    family_by_threads(use_std=True)
if PLOT_SPEEDUP:
    plot_speedup()

n_pdf = sum(len(files) for _, _, files in os.walk(OUTDIR))
print(f"[OK] Graficos gerados em {OUTDIR}/ ({n_pdf} PDFs).")
print(f"[INFO] IC: {'t de Student' if HAVE_SCIPY else 'aproximacao normal 1,96 (scipy ausente)'}")
