#!/bin/bash
#
# Helper script to submit MLAAD Stage 2 (intermediate) training jobs with different seeds,
# processors, and classifiers.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train_seed_template.sh"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

# SEEDS=(42 123 222)
SEEDS=(42 123 222)
# SEEDS=(42)
DATASET="MLAADIntermediateDataset_pair"
WALLTIME="30:00:00"
PROCESSORS=(AASIST MHFA)
# PROCESSORS=(AASIST)
# CLASSIFIERS=(FFDiff FFCosine FFConcat3 FFAttn2)
CLASSIFIERS=(FFDiffAbs FFDiffQuadratic)
# CLASSIFIERS=(FFAttn2)
# CLASSIFIERS=(FFConcat3)

for PROCESSOR in "${PROCESSORS[@]}"; do
  for CLASSIFIER in "${CLASSIFIERS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      JOB_SCRIPT="${SCRIPT_DIR}/train_seed_intermediate_${PROCESSOR}_${CLASSIFIER}_${SEED}.sh"
      sed -e "s/{{SEED}}/${SEED}/g" \
          -e "s/{{DATASET}}/${DATASET}/g" \
          -e "s/{{WALLTIME}}/${WALLTIME}/g" \
          -e "s/{{PROCESSOR}}/${PROCESSOR}/g" \
          -e "s/{{CLASSIFIER}}/${CLASSIFIER}/g" \
          "${TEMPLATE}" > "${JOB_SCRIPT}"
      chmod +x "${JOB_SCRIPT}"
      echo "Submitting Stage 2 intermediate on Karolina (${PROCESSOR}+${CLASSIFIER}) with seed ${SEED} (walltime ${WALLTIME})..."
      sbatch "${JOB_SCRIPT}"
    done
  done
done
