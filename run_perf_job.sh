#!/bin/bash
#
# run_perf.sh - Benchmarking reprodutível de algoritmos de ordenação com perf.
#
# Correcoes desta versao (em relacao a multiplexacao observada):
#   - NMI watchdog desligado: devolve 1 contador de hardware que o kernel
#     consome permanentemente (era por isso que ate cycles:u multiplexava).
#   - Eventos divididos em passagens com no maximo 4 eventos de proposito
#     geral cada (instructions/cycles ficam nos contadores FIXOS), de modo
#     que NENHUMA passagem multiplexa. Sem multiplexacao, sem estimativa:
#     instructions:u e branches:u saem EXATAMENTE constantes (validacao).
#   - NOVO: passagem P3 de ENERGIA via RAPL (power/energy-pkg/ e
#     power/energy-ram/). RAPL e por-socket, NAO usa o modificador :u e NAO
#     multiplexa com a PMU, entao P3 nao afeta a exatidao de P1/P2. Os eventos
#     disponiveis sao DETECTADOS; se 'energy-ram' nao existir, usa so 'pkg';
#     se nenhum existir, a energia e pulada.
#
# Demais controles de reprodutibilidade:
#   Camada 1 (kernel atribuido a task): modificador :u em todos os eventos PMU.
#   Camada 2 (ASLR / layout de pilha):  setarch -R.
#   Pinagem de core:                    taskset (serial -> 1 core).
#   Ambiente constante:                 mesmos args/env entre execucoes.
#
# NOTA sobre energia: RAPL mede o SOCKET inteiro (nao o processo). Para numeros
# confiaveis, rode com a maquina o mais ociosa possivel e idealmente no no
# dedicado. A energia so e significativa nas entradas GRANDES (execucao de
# varios ms+); nos casos pequenos ela e dominada por ruido/startup.
#
set -euo pipefail

# -----------------------------------------------------------------------------
# Argumentos
# -----------------------------------------------------------------------------
if [ $# -lt 3 ]; then
    echo "Uso: $0 <binary> <input> <threads> [nruns]"
    echo "  nruns default = 100 (use 10 para testar)"
    exit 1
fi

BIN="$1"
INPUT="$2"
THREADS="$3"
NRUNS="${4:-100}"

WORKDIR="$(pwd)"
cd "$WORKDIR"

BIN_NAME=$(basename "$BIN")
INPUT_NAME=$(basename "$INPUT")

# -----------------------------------------------------------------------------
# Checagens
# -----------------------------------------------------------------------------
if ! command -v perf &> /dev/null;    then echo "[ERRO] perf nao encontrado.";    exit 1; fi
if ! command -v setarch &> /dev/null; then echo "[ERRO] setarch nao encontrado."; exit 1; fi
if ! command -v taskset &> /dev/null; then echo "[ERRO] taskset nao encontrado."; exit 1; fi
if [ ! -x "$BIN" ];   then echo "[ERRO] binario nao executavel: $BIN"; exit 1; fi
if [ ! -f "$INPUT" ]; then echo "[ERRO] entrada nao encontrada: $INPUT"; exit 1; fi

mkdir -p results
PERF="sudo perf"

# -----------------------------------------------------------------------------
# Libera um contador: desliga o NMI watchdog (restaura no fim via trap).
# Era a causa de cycles:u aparecer a ~63% mesmo sendo contador fixo.
# -----------------------------------------------------------------------------
NMI_ORIG="$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null || echo 1)"
restore_nmi() {
    sudo sysctl -q -w "kernel.nmi_watchdog=${NMI_ORIG}" 2>/dev/null || true
}
trap restore_nmi EXIT
sudo sysctl -q -w kernel.nmi_watchdog=0 2>/dev/null \
    || echo "[WARN] nao foi possivel desligar o NMI watchdog (segue assim mesmo)."

# -----------------------------------------------------------------------------
# Deteccao dos dominios de energia RAPL disponiveis.
# Testa cada evento com um 'true' rapido; mantem so os que o perf suporta.
# -----------------------------------------------------------------------------
probe_event() {
    local ev="$1" out
    out=$($PERF stat -e "$ev" -- true 2>&1) || return 1
    echo "$out" | grep -qiE "not supported|<not |not counted|invalid|cannot|unknown" && return 1
    return 0
}

ENERGY=""
for ev in "power/energy-pkg/" "power/energy-ram/"; do
    if probe_event "$ev"; then
        ENERGY="${ENERGY:+$ENERGY,}$ev"
    fi
done

# -----------------------------------------------------------------------------
# Wrappers de controle
# -----------------------------------------------------------------------------
SETARCH="setarch $(uname -m) -R"          # Camada 2: ASLR off

if [ "$THREADS" -eq 1 ]; then
    PIN="taskset -c 0"
else
    PIN="taskset -c 0-$((THREADS - 1))"   # CAVEAT: pode incluir irmaos SMT
    export OMP_NUM_THREADS="$THREADS"
    export OMP_PROC_BIND="close"
    export OMP_PLACES="cores"
    export OMP_WAIT_POLICY="passive"      # mata spin-wait das barreiras
fi

# -----------------------------------------------------------------------------
# Eventos divididos em passagens MUX-FREE (<= 4 eventos de proposito geral).
# instructions e cycles usam contadores FIXOS, nao contam no limite de 4.
#
#   P1: instructions, cycles  (fixos) + branches, branch-misses,
#                                        cache-references, cache-misses (4 GP)
#   P2: instructions, cycles  (fixos) + L1-dcache-load-misses, LLC-load-misses,
#                                        dTLB-load-misses, iTLB-load-misses (4 GP)
#   P3: ENERGIA (RAPL)         -> power/energy-pkg/ [, power/energy-ram/]
#                                 sem :u, sem multiplexar com a PMU.
#
# instructions:u aparece em P1 e P2 (ancora de cross-check).
# branches:u na P1 e o evento-prova de determinismo: deve sair constante.
# -----------------------------------------------------------------------------
P1="instructions:u,cycles:u,branches:u,branch-misses:u,cache-references:u,cache-misses:u"
P2="instructions:u,cycles:u,L1-dcache-load-misses:u,LLC-load-misses:u,dTLB-load-misses:u,iTLB-load-misses:u"

# -----------------------------------------------------------------------------
# Cabecalho
# -----------------------------------------------------------------------------
echo "[INFO] Binario     : $BIN_NAME"
echo "[INFO] Entrada     : $INPUT_NAME"
echo "[INFO] Threads     : $THREADS"
echo "[INFO] Repeticoes  : $NRUNS"
echo "[INFO] Pin         : $PIN"
echo "[INFO] ASLR        : off (setarch -R)"
echo "[INFO] NMI watchdog: off (era ${NMI_ORIG})"
echo "[INFO] Passagens   : P1=$P1"
echo "[INFO]               P2=$P2"
if [ -n "$ENERGY" ]; then
    echo "[INFO]               P3=$ENERGY (energia RAPL, em Joules)"
else
    echo "[WARN] RAPL indisponivel (sem power/energy-*): passagem de energia DESATIVADA."
fi
echo

FAILED=0

# -----------------------------------------------------------------------------
# Loop: 2 (ou 3) passagens por run. P1/P2 mux-free; P3 = energia (se houver).
# -----------------------------------------------------------------------------
for i in $(seq 1 "$NRUNS"); do
    RUN=$(printf "%03d" "$i")
    BASE="results/${BIN_NAME}_${INPUT_NAME}_t${THREADS}_run${RUN}"

    echo "[INFO] Run ${RUN}/${NRUNS}"

    if ! $PERF stat -e "$P1" -o "${BASE}.p1" \
        -- $PIN $SETARCH "$BIN" "$INPUT" "$THREADS"; then
        echo "[WARN] Run ${RUN} P1 falhou."; FAILED=$((FAILED + 1)); continue
    fi

    if ! $PERF stat -e "$P2" -o "${BASE}.p2" \
        -- $PIN $SETARCH "$BIN" "$INPUT" "$THREADS"; then
        echo "[WARN] Run ${RUN} P2 falhou."; FAILED=$((FAILED + 1)); continue
    fi

    if [ -n "$ENERGY" ]; then
        if ! $PERF stat -e "$ENERGY" -o "${BASE}.p3" \
            -- $PIN $SETARCH "$BIN" "$INPUT" "$THREADS"; then
            echo "[WARN] Run ${RUN} P3 (energia) falhou."; FAILED=$((FAILED + 1)); continue
        fi
    fi
done

echo
echo "[INFO] Concluido: $((NRUNS - FAILED))/${NRUNS} runs OK, ${FAILED} com falha."
echo
echo "[CHECK] Nenhuma saida P1/P2 deve conter '%' entre parenteses (= sem multiplexacao):"
echo "        grep -H '%)' results/${BIN_NAME}_${INPUT_NAME}_t${THREADS}_run*.p1 results/*.p2 || echo '  OK: nenhum evento multiplexado'"
echo "[CHECK] instructions:u e branches:u devem ser identicos em todos os runs:"
echo "        grep -h 'instructions:u' results/${BIN_NAME}_${INPUT_NAME}_t${THREADS}_run*.p1"
echo "        grep -h 'branches:u'     results/${BIN_NAME}_${INPUT_NAME}_t${THREADS}_run*.p1"
if [ -n "$ENERGY" ]; then
echo "[CHECK] energia (Joules) por run:"
echo "        grep -h 'Joules' results/${BIN_NAME}_${INPUT_NAME}_t${THREADS}_run*.p3"
fi
