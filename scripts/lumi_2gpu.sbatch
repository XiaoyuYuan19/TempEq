#!/bin/bash
#SBATCH --job-name=tempeq
#SBATCH --account=project_462001364
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# TempEq 2-GPU DDP launcher on LUMI.
#
# Usage:
#   CONFIG=clip_pf2 sbatch scripts/lumi_2gpu.sbatch
#
# Any env var defined in configs/<CONFIG>.env can be overridden inline:
#   CONFIG=clip_pf2 EPOCHS=50 BATCH_SIZE=32 sbatch scripts/lumi_2gpu.sbatch
#
# Required by the runtime: TempEq files at the project root, UCF101 tar at
# $PROJECT_DIR/cache/UCF-101.tar (extracted into /tmp per-job), and the
# standard split files reachable via SPLIT_DIR below.

PROJECT_DIR="/scratch/project_462001364/$USER/TempEq"
cd "$PROJECT_DIR"

# Load a named config from configs/ if CONFIG=name is set. Inline env
# vars (e.g. EPOCHS=2 sbatch ...) take precedence over the config file:
# values in the config are only applied when the variable is not already
# set in the calling environment.
CONFIG="${CONFIG:-}"
if [ -n "$CONFIG" ]; then
    CFG_PATH="$PROJECT_DIR/configs/${CONFIG}.env"
    if [ ! -f "$CFG_PATH" ]; then
        echo "ERROR: config file not found: $CFG_PATH" >&2
        exit 1
    fi
    while IFS='=' read -r key value; do
        # skip comments and blank lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${key// }" ]] && continue
        # strip whitespace around key
        key="${key// }"
        # only export if not already set in the environment
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$CFG_PATH"
fi

# Defaults (apply when neither config nor inline override is given).
RUN_NAME="${RUN_NAME:-tempeq_run}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WINDOW="${WINDOW:-5}"
SAMPLER="${SAMPLER:-geometric}"
GEOM_P="${GEOM_P:-0.008}"
MIN_K="${MIN_K:-1}"
MAX_K="${MAX_K:-150}"
PAIRS_PER_VIDEO="${PAIRS_PER_VIDEO:-1}"
AUG_MODE="${AUG_MODE:-ssl}"
LOSS_MODE="${LOSS_MODE:-pf2}"
PF2_DUAL_LR="${PF2_DUAL_LR:-0.01}"
PF2_LAM_INIT="${PF2_LAM_INIT:-0.0}"
PF2_LAM_MAX="${PF2_LAM_MAX:-0.0}"
VICREG_SIM="${VICREG_SIM:-25.0}"
VICREG_VAR="${VICREG_VAR:-25.0}"
VICREG_COV="${VICREG_COV:-1.0}"
PROJ_DIM="${PROJ_DIM:-128}"
PROJ_HIDDEN="${PROJ_HIDDEN:-512}"
PROJ_LAYERS="${PROJ_LAYERS:-2}"
PROJ_BN="${PROJ_BN:-0}"
CLIP_LEN="${CLIP_LEN:-1}"
CLIP_STRIDE="${CLIP_STRIDE:-1}"
OPTIMIZER="${OPTIMIZER:-lars}"
LR="${LR:-0.2}"
WD="${WD:-1e-6}"
GRAD_CLIP="${GRAD_CLIP:-0}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PROBE_EVERY="${PROBE_EVERY:-5}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
PROBE_BATCH="${PROBE_BATCH:-32}"
RESUME="${RESUME:-}"

LOGDIR="logs/${RUN_NAME}"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/train.log"
exec >> "$LOGFILE" 2>&1

echo "========== Job $SLURM_JOB_ID started: $(date) =========="
echo "host=$(hostname)  partition=$SLURM_JOB_PARTITION  nodes=$SLURM_JOB_NUM_NODES"
echo "ntasks=$SLURM_NTASKS  gpus_per_node=$SLURM_GPUS_ON_NODE"
echo "CONFIG=$CONFIG RUN=$RUN_NAME LOSS=$LOSS_MODE SAMPLER=$SAMPLER CLIP_LEN=$CLIP_LEN STRIDE=$CLIP_STRIDE AUG=$AUG_MODE"
echo "PROJ=${PROJ_DIM}/${PROJ_HIDDEN}/${PROJ_LAYERS}/BN=${PROJ_BN} OPT=$OPTIMIZER LR=$LR WD=$WD EPOCHS=$EPOCHS BS=$BATCH_SIZE"

module use /appl/local/csc/modulefiles
module load pytorch/2.7

set -euo pipefail

TAR="$PROJECT_DIR/cache/UCF-101.tar"
echo "[$(date +%H:%M:%S)] extracting tar..."
tar xf "$TAR" -C /tmp/
echo "[$(date +%H:%M:%S)] staged ($(ls /tmp/UCF-101 | wc -l) classes)"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29502}"
export PYTHONUNBUFFERED=1
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

VIDEO_DIR="/tmp/UCF-101"
SPLIT_DIR="${SPLIT_DIR:-$PROJECT_DIR/data/ucfTrainTestlist}"

srun bash -c "
    export MIOPEN_USER_DB_PATH=/tmp/${SLURM_JOB_ID}/miopen_\$SLURM_LOCALID
    export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH
    mkdir -p \$MIOPEN_USER_DB_PATH
    python3 train.py \
        --video-root '$VIDEO_DIR' \
        --split-root '$SPLIT_DIR' \
        --out-dir '$LOGDIR' \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --optimizer $OPTIMIZER \
        --weight-decay $WD \
        --grad-clip $GRAD_CLIP \
        --window $WINDOW \
        --sampler $SAMPLER \
        --geometric-p $GEOM_P \
        --min-k $MIN_K \
        --max-k $MAX_K \
        --pairs-per-video $PAIRS_PER_VIDEO \
        --aug-mode $AUG_MODE \
        --loss $LOSS_MODE \
        --pf2-dual-lr $PF2_DUAL_LR \
        --pf2-lam-init $PF2_LAM_INIT \
        --pf2-lam-max $PF2_LAM_MAX \
        --vicreg-sim $VICREG_SIM \
        --vicreg-var $VICREG_VAR \
        --vicreg-cov $VICREG_COV \
        --proj-dim $PROJ_DIM \
        --proj-hidden $PROJ_HIDDEN \
        --proj-layers $PROJ_LAYERS \
        \$([ \"$PROJ_BN\" = \"1\" ] && echo --proj-bn) \
        --clip-len $CLIP_LEN \
        --clip-stride $CLIP_STRIDE \
        ${RESUME:+--resume $RESUME} \
        --temperature 0.1 \
        --workers 6 \
        --save-every $SAVE_EVERY \
        --probe-every $PROBE_EVERY \
        --probe-epochs $PROBE_EPOCHS \
        --probe-batch $PROBE_BATCH
"

echo "========== Job $SLURM_JOB_ID ended: $(date) =========="
