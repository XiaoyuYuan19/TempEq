#!/usr/bin/env bash
# Helper to prepare UCF101 for TempEq training.
#
# Steps:
#   1. Download UCF101 videos + train/test split files.
#   2. Extract the videos and pack into a single tar for fast staging on
#      shared filesystems (Lustre on LUMI hates many small files).
#
# After running, you should have:
#   $DATA_ROOT/UCF-101/<ClassName>/v_*.avi
#   $DATA_ROOT/ucfTrainTestlist/{classInd.txt,trainlist01.txt,testlist01.txt}
#   $DATA_ROOT/UCF-101.tar    (single-file archive for fast extraction)
#
# Adjust DATA_ROOT to taste.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

VIDEOS_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
SPLITS_URL="https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip"

if [ ! -d UCF-101 ]; then
    echo "[1/3] Downloading and extracting UCF101 videos..."
    wget -nc "$VIDEOS_URL"
    if ! command -v unrar >/dev/null 2>&1; then
        echo "ERROR: unrar is required to extract UCF101.rar" >&2
        echo "  Ubuntu/Debian:  sudo apt install unrar" >&2
        echo "  conda:          conda install -c conda-forge unrar" >&2
        exit 1
    fi
    unrar x -y UCF101.rar
fi

if [ ! -d ucfTrainTestlist ]; then
    echo "[2/3] Downloading and extracting split files..."
    wget -nc "$SPLITS_URL"
    unzip -o UCF101TrainTestSplits-RecognitionTask.zip
fi

if [ ! -f UCF-101.tar ]; then
    echo "[3/3] Packing videos into UCF-101.tar (faster staging on Lustre)..."
    tar cf UCF-101.tar UCF-101
fi

echo
echo "Done. Set --video-root and --split-root for train.py:"
echo "  --video-root $DATA_ROOT/UCF-101"
echo "  --split-root $DATA_ROOT/ucfTrainTestlist"
echo
echo "On LUMI, copy UCF-101.tar to your project cache and extract on each node:"
echo "  cp $DATA_ROOT/UCF-101.tar /scratch/project_xxx/\$USER/TempEq/cache/"
echo "  # then in sbatch:  tar xf UCF-101.tar -C /tmp/"
