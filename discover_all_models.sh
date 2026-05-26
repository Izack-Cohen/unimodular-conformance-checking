#!/bin/bash

DATA_DIR="data"
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

DATASETS=(
    "BPI_Challenge_2012.xes"
    "BPI_Challenge_2017.xes"
    "BPI_Challenge_2019.xes"
    "Sepsis_cases.xes"
    "BPI_Challenge_2013_incidents.xes"
)

echo "========================================================================"
echo "BATCH MODEL DISCOVERY - FULL LOG EVALUATION"
echo "========================================================================"
echo "Processing ${#DATASETS[@]} datasets"
echo "Results will be saved to: $RESULTS_DIR"
echo "Evaluation mode: FULL LOG (no sampling)"
echo "========================================================================"

START_TIME=$(date +%s)

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    dataset_num=$((i + 1))

    echo ""
    echo "========================================================================"
    echo "[$dataset_num/${#DATASETS[@]}] Processing: $dataset"
    echo "========================================================================"

    dataset_path="$DATA_DIR/$dataset"

    if [ ! -f "$dataset_path" ]; then
        echo "❌ ERROR: Dataset not found: $dataset_path"
        echo "   Skipping..."
        continue
    fi

    echo ""
    echo "Starting model discovery..."
    echo "Command: python discover_model_variants.py $dataset_path --output-dir $RESULTS_DIR"
    echo ""

    python discover_model_variants.py \
        "$dataset_path" \
        --output-dir "$RESULTS_DIR"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS: Model discovery completed for $dataset"
    else
        echo ""
        echo "❌ ERROR: Model discovery failed for $dataset"
    fi

    echo ""
    echo "Progress: $dataset_num/${#DATASETS[@]} datasets completed"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================================================"
echo "BATCH DISCOVERY COMPLETE"
echo "========================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary files created:"
for dataset in "${DATASETS[@]}"; do
    dataset_name="${dataset%.xes}"
    summary_file="$RESULTS_DIR/${dataset_name}_discovered_models_summary.csv"
    if [ -f "$summary_file" ]; then
        num_models=$(tail -n +2 "$summary_file" | wc -l)
        echo "  ✓ $dataset_name: $num_models models"
    else
        echo "  ✗ $dataset_name: No summary found"
    fi
done
echo ""
echo "Next step: Run experiments with discovered models using:"
echo "  python run_model_variant_experiments.py data/<dataset>.xes $RESULTS_DIR"
echo "========================================================================"