#!/bin/bash
# Hyperparameter sweep script for cloud training
# Usage: ./sweep_hyperparams.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dequant Hyperparameter Sweep${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "src/__main__.py" ]; then
    echo -e "${RED}Error: Must run from dequant project root directory${NC}"
    exit 1
fi

# Configuration
PYTHON_CMD="python -m src"
CHECKPOINT_BASE_DIR=".data/checkpoints"
TENSORBOARD_BASE_DIR="runs"

# Create a timestamp for this sweep
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
echo -e "${YELLOW}Sweep ID: ${SWEEP_ID}${NC}"
echo ""

# Define experiments as an array
# Format: "experiment_name|d_model|n_heads|n_layers|dropout|learning_rate|batch_size|num_epochs|warmup_epochs"
declare -a EXPERIMENTS=(
    # d_model=256, n_heads=8
    # larger models
    "LR_warmup_d256_h8_l4|256|8|4|0.0|1e-4|128|4|1"
)

# Optional: Limit number of epochs for quick testing
# Uncomment the line below and adjust QUICK_EPOCHS to run shorter experiments
QUICK_EPOCHS=4

echo -e "${GREEN}Total experiments planned: ${#EXPERIMENTS[@]}${NC}"
echo ""

# Ask for confirmation
read -p "Start hyperparameter sweep? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Experiments${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Track experiment results
declare -a RESULTS=()

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"

    # Parse experiment parameters
    IFS='|' read -r NAME D_MODEL N_HEADS N_LAYERS DROPOUT LR BATCH_SIZE EPOCHS WARMUP_EPOCHS <<< "$EXP"

    # Override epochs for quick testing if QUICK_EPOCHS is set
    if [ ! -z "$QUICK_EPOCHS" ]; then
        EPOCHS=$QUICK_EPOCHS
    fi

    EXP_NUM=$((i + 1))
    TOTAL_EXPS=${#EXPERIMENTS[@]}

    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Experiment ${EXP_NUM}/${TOTAL_EXPS}: ${NAME}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo -e "d_model:       ${D_MODEL}"
    echo -e "n_heads:       ${N_HEADS}"
    echo -e "n_layers:      ${N_LAYERS}"
    echo -e "dropout:       ${DROPOUT}"
    echo -e "learning_rate: ${LR}"
    echo -e "batch_size:    ${BATCH_SIZE}"
    echo -e "num_epochs:    ${EPOCHS}"
    echo -e "warmup_epochs: ${WARMUP_EPOCHS}"
    echo ""

    # Create experiment-specific checkpoint directory
    EXP_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/sweep_${SWEEP_ID}/${NAME}"
    mkdir -p "$EXP_CHECKPOINT_DIR"

    # Build the training command
    TRAIN_CMD="${PYTHON_CMD} \
        --config.model.transformer.d-model=${D_MODEL} \
        --config.model.transformer.n-heads=${N_HEADS} \
        --config.model.transformer.n-layers=${N_LAYERS} \
        --config.model.transformer.dropout=${DROPOUT} \
        --config.train.learning-rate=${LR} \
        --config.train.batch-size=${BATCH_SIZE} \
        --config.train.num-epochs=${EPOCHS} \
        --config.train.lr-warmup-epochs=${WARMUP_EPOCHS} \
        --config.train.run-name=${NAME} \
        --config.train.checkpoint-dir=${EXP_CHECKPOINT_DIR} \
        --config.train.max-train-samples=25000 \
        --config.train.max-val-samples=5000 \
        --config.train.no-auto-preprocess \
        train"

    # Log the command
    echo -e "${GREEN}Running:${NC} ${TRAIN_CMD}"
    echo ""

    # Run the training
    START_TIME=$(date +%s)

    if $TRAIN_CMD; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        DURATION_MIN=$((DURATION / 60))

        echo ""
        echo -e "${GREEN}✓ Experiment ${NAME} completed successfully!${NC}"
        echo -e "${GREEN}  Duration: ${DURATION_MIN} minutes${NC}"
        RESULTS+=("${EXP_NUM}. ${NAME}: SUCCESS (${DURATION_MIN}m)")
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        DURATION_MIN=$((DURATION / 60))

        echo ""
        echo -e "${RED}✗ Experiment ${NAME} failed!${NC}"
        echo -e "${RED}  Duration: ${DURATION_MIN} minutes${NC}"
        RESULTS+=("${EXP_NUM}. ${NAME}: FAILED (${DURATION_MIN}m)")

        # Ask if we should continue
        read -p "Continue with remaining experiments? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping sweep."
            break
        fi
    fi

    # Short pause between experiments
    sleep 2
    # Clear MPS cache between experiments
    python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None" 2>/dev/null || true

done

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Sweep Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Results Summary:${NC}"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo -e "${YELLOW}Checkpoints saved in:${NC} ${CHECKPOINT_BASE_DIR}/sweep_${SWEEP_ID}/"
echo ""
echo -e "${GREEN}To compare results with TensorBoard:${NC}"
echo "  tensorboard --logdir ${TENSORBOARD_BASE_DIR}"
echo ""
echo -e "${GREEN}To find the best checkpoint:${NC}"
echo "  ls -lh ${CHECKPOINT_BASE_DIR}/sweep_${SWEEP_ID}/*/checkpoint_*.pt"
echo ""
