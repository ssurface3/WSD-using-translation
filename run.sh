#!/bin/bash

# ==============================================================================
# WSD Pipeline Launcher
# Usage: ./run.sh <path_to_input_data.tsv>
# ==============================================================================

PYTHON_EXEC="python3"  
ENTRY_POINT="run_pipe.py"


if [ "$#" -ne 1 ]; then
    echo "Error: Illegal number of parameters."
    echo "Usage: $0 <path_to_dataset.tsv>"
    exit 1
fi

INPUT_FILE=$1


if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi


if command -v nvidia-smi &> /dev/null; then
    echo "System: NVIDIA GPU detected. Pipeline will utilize CUDA."
else
    echo "System: No NVIDIA GPU detected. Pipeline will run on CPU (slower)."
fi


echo "Starting WSD Pipeline with input: $INPUT_FILE"
echo "------------------------------------------------"
$PYTHON_EXEC "$ENTRY_POINT" "$INPUT_FILE"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Pipeline completed successfully. Check 'checkpoint_translated.tsv'."
else
    echo "Pipeline failed with error code $exit_code."
fi