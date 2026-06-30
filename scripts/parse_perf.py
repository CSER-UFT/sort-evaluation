import re
import glob
import os
import pandas as pd

# -----------------------------------------------------------------------------
# Normaliza numeros no formato PT/BR -> float
# -----------------------------------------------------------------------------
def parse_number(s):
    s = s.strip()
    if not s:
        return 0.0
    s = s.replace(" ", "")
    if ',' in s:
        s = s.replace('.', '')
        s = s.replace(',', '.')
    else:
        s = s.replace('.', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


# Extrai uma metrica do texto do perf (ancorado em inicio de linha; tolera :u).
def get_metric(name, text):
    pattern = rf"^\s*([\d.,]+)\s+{re.escape(name)}(?::u)?\b"
    m = re.search(pattern, text, re.MULTILINE)
    return parse_number(m.group(1)) if m else None


def get_energy(name, text):
    """Le um evento RAPL no formato '1.234,56 Joules power/energy-pkg/'.
    A unidade 'Joules' aparece ENTRE o numero e o nome do evento, por isso
    precisa de um padrao proprio (o get_metric nao captura)."""
    pattern = rf"^\s*([\d.,]+)\s+Joules\s+{re.escape(name)}"
    m = re.search(pattern, text, re.MULTILINE)
    return parse_number(m.group(1)) if m else None


MUX_RE = re.compile(r"\(\s*\d+[.,]\d+\s*%\s*\)")


def input_num(s):
    """Chave numerica para ordenar entradas: '100000', '5000000.in', 'input2'..."""
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0


# -----------------------------------------------------------------------------
# Regex do nome do arquivo (sem a extensao .p1/.p2):
#   <algo>_<mode>_<input>(.ext)?_t<threads>_run<NNN>
# <input> aceita nomes numericos novos (100000, 5000000, ...) E antigos (input1).
# Extensao (qualquer) opcional; run opcional.
# -----------------------------------------------------------------------------
NAME_RE = re.compile(
    r"^(?P<algo>.+?)_(?P<mode>serial|parallel)_(?P<input>(?:input)?\d+)(?:\.[A-Za-z0-9]+)?"
    r"_t(?P<threads>\d+)(?:_run(?P<run>\d+))?$"
)

P1_EVENTS = {
    "instructions":     "instructions",
    "cycles":           "cycles",
    "branches":         "branches",
    "branch_misses":    "branch-misses",
    "cache_references": "cache-references",
    "cache_misses":     "cache-misses",
}
P2_EVENTS = {
    "instructions":          "instructions",
    "cycles":                "cycles",
    "L1_dcache_load_misses": "L1-dcache-load-misses",
    "LLC_load_misses":       "LLC-load-misses",
    "dTLB_load_misses":      "dTLB-load-misses",
    "iTLB_load_misses":      "iTLB-load-misses",
}

os.makedirs("results", exist_ok=True)

records = {}
files = sorted(glob.glob("results/*.p1")) + sorted(glob.glob("results/*.p2")) \
        + sorted(glob.glob("results/*.p3"))
if not files:
    raise SystemExit("[ERRO] Nenhum arquivo .p1/.p2 encontrado em results/")

ignored = 0
for file in files:
    stem, ext = os.path.splitext(os.path.basename(file))
    match = NAME_RE.match(stem)
    if not match:
        print(f"[AVISO] Nome invalido, ignorado: {stem}{ext}")
        ignored += 1
        continue

    algo = match.group("algo")
    mode = match.group("mode")
    inp = match.group("input")
    threads = int(match.group("threads"))
    run = int(match.group("run")) if match.group("run") else 1

    key = (algo, inp, mode, threads, run)
    rec = records.setdefault(key, {
        "algo": algo, "input": inp, "mode": mode,
        "threads": threads, "run": run,
    })

    with open(file, "r", encoding="utf-8") as f:
        txt = f.read()

    if MUX_RE.search(txt):
        print(f"[AVISO] Multiplexacao detectada em {os.path.basename(file)} "
              f"(contagem estimada, nao exata).")

    if ext == ".p3":
        # Passagem de energia (RAPL). Sem eventos de PMU; so Joules + tempo.
        rec["energy_pkg"] = get_energy("power/energy-pkg/", txt)
        rec["energy_ram"] = get_energy("power/energy-ram/", txt)
        rec["_time.p3"] = get_metric("seconds time elapsed", txt)
        continue

    evmap = P1_EVENTS if ext == ".p1" else P2_EVENTS
    for col, ev in evmap.items():
        v = get_metric(ev, txt)
        if v is not None:
            rec.setdefault(col, v)

    rec[f"_instr{ext}"] = get_metric("instructions", txt)
    rec[f"_time{ext}"] = get_metric("seconds time elapsed", txt)

if not records:
    raise SystemExit(f"[ERRO] Nenhum arquivo casou com o padrao de nome "
                     f"(<algo>_<serial|parallel>_<input>[.ext]_t<n>_run<NNN>). "
                     f"{ignored} ignorado(s).")


def num(rec, k):
    v = rec.get(k)
    return v if v is not None else 0.0

rows = []
for key, rec in records.items():
    instr   = num(rec, "instructions")
    cycles  = num(rec, "cycles")
    branches = num(rec, "branches")
    br_miss  = num(rec, "branch_misses")
    cref     = num(rec, "cache_references")
    cmiss    = num(rec, "cache_misses")

    i1, i2 = rec.get("_instr.p1"), rec.get("_instr.p2")
    if i1 and i2 and abs(i1 - i2) / max(i1, i2) > 0.001:
        print(f"[AVISO] instructions difere entre passagens em "
              f"{rec['algo']} {rec['input']} {rec['mode']} t{rec['threads']} "
              f"run{rec['run']:03d}: p1={i1:.0f} p2={i2:.0f}")

    times = [t for t in (rec.get("_time.p1"), rec.get("_time.p2")) if t]
    time_sec = sum(times) / len(times) if times else 0.0
    stalled_be = 0.0

    # Energia (RAPL). pkg e ram podem faltar; total = soma do que existir.
    e_pkg = num(rec, "energy_pkg")
    e_ram = num(rec, "energy_ram")
    e_total = e_pkg + e_ram
    # Potencia media (W = J/s), EDP (J*s) e ED2P (J*s^2) usando o tempo medido.
    power_w = e_total / time_sec if (e_total and time_sec) else 0.0
    edp     = e_total * time_sec
    ed2p    = e_total * time_sec * time_sec

    rows.append({
        "algo": rec["algo"], "input": rec["input"], "mode": rec["mode"],
        "threads": rec["threads"], "run": rec["run"],
        "instructions": instr, "cycles": cycles,
        "cpi": cycles / instr if instr else 0.0,
        "ipc": instr / cycles if cycles else 0.0,
        "branch_miss_rate": br_miss / branches if branches else 0.0,
        "cache_miss_rate": cmiss / cref if cref else 0.0,
        "stalled_be_ratio": stalled_be / cycles if cycles else 0.0,
        "L1_dcache_load_misses": num(rec, "L1_dcache_load_misses"),
        "LLC_load_misses": num(rec, "LLC_load_misses"),
        "dTLB_load_misses": num(rec, "dTLB_load_misses"),
        "iTLB_load_misses": num(rec, "iTLB_load_misses"),
        "time_sec": time_sec,
        "energy_pkg_joules": e_pkg,
        "energy_ram_joules": e_ram,
        "energy_total_joules": e_total,
        "power_watts": power_w,
        "edp": edp,
        "ed2p": ed2p,
    })

df_raw = pd.DataFrame(rows)
df_raw["_inp_n"] = df_raw["input"].map(input_num)
df_raw = df_raw.sort_values(
    ["algo", "_inp_n", "mode", "threads", "run"]
).drop(columns="_inp_n").reset_index(drop=True)
df_raw.to_csv("results/results_raw.csv", index=False)
print(f"[OK] {len(df_raw)} execucoes lidas -> results/results_raw.csv")

group_keys = ["algo", "input", "mode", "threads"]
metric_cols = [
    "instructions", "cycles", "cpi", "ipc",
    "branch_miss_rate", "cache_miss_rate", "stalled_be_ratio",
    "L1_dcache_load_misses", "LLC_load_misses",
    "dTLB_load_misses", "iTLB_load_misses", "time_sec",
    "energy_pkg_joules", "energy_ram_joules", "energy_total_joules",
    "power_watts", "edp", "ed2p",
]

agg = df_raw.groupby(group_keys)[metric_cols].agg(["mean", "std"])
agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
agg = agg.reset_index()

nruns = df_raw.groupby(group_keys).size().rename("n_runs").reset_index()
agg = agg.merge(nruns, on=group_keys)

agg["time_cv"] = agg.apply(
    lambda r: (r["time_sec_std"] / r["time_sec_mean"]) if r["time_sec_mean"] else 0.0,
    axis=1
)

serial_mean = {
    (r["algo"], r["input"]): r["time_sec_mean"]
    for _, r in agg.iterrows()
    if r["mode"] == "serial" and r["threads"] == 1
}

def speedup(r):
    base = serial_mean.get((r["algo"], r["input"]))
    if base and r["time_sec_mean"]:
        return base / r["time_sec_mean"]
    return None

agg["speedup"] = agg.apply(speedup, axis=1)
agg["efficiency"] = agg.apply(
    lambda r: (r["speedup"] / r["threads"]) if r["speedup"] else None, axis=1)

def bottleneck(r):
    b = []
    if r["ipc_mean"] < 1.0 or r["stalled_be_ratio_mean"] > 0.2:
        b.append("ILP/pipeline")
    if r["branch_miss_rate_mean"] > 0.05:
        b.append("Branch")
    if r["cache_miss_rate_mean"] > 0.01 or r["LLC_load_misses_mean"] > 0:
        b.append("Memory/cache")
    if r["dTLB_load_misses_mean"] > 0 or r["iTLB_load_misses_mean"] > 0:
        b.append("TLB")
    return ", ".join(b)

agg["bottleneck"] = agg.apply(bottleneck, axis=1)

front = group_keys + ["n_runs",
                      "time_sec_mean", "time_sec_std", "time_cv",
                      "speedup", "efficiency",
                      "ipc_mean", "ipc_std", "cpi_mean", "cpi_std",
                      "branch_miss_rate_mean", "cache_miss_rate_mean"]
rest = [c for c in agg.columns if c not in front and c != "bottleneck"]
agg = agg[front + rest + ["bottleneck"]]

agg["_inp_n"] = agg["input"].map(input_num)
agg = agg.sort_values(["algo", "_inp_n", "mode", "threads"]).drop(
    columns="_inp_n").reset_index(drop=True)
agg.to_csv("results/results_summary.csv", index=False)
print(f"[OK] {len(agg)} casos agregados -> results/results_summary.csv")
