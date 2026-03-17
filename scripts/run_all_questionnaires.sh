#!/usr/bin/env bash
# Run all runnable questionnaires for canonical patients only.
#
# Usage:
#   ./scripts/run_all_questionnaires.sh              # skip already-completed pairs
#   ./scripts/run_all_questionnaires.sh --force      # re-run and overwrite all results
#
# The script must be run from the repository root:
#   bash scripts/run_all_questionnaires.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_questionnaire.py"

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

# Discover patients and questionnaires from the data directory
PATIENTS_DIR="$ROOT_DIR/data/patients"
QUESTIONNAIRES_DIR="$ROOT_DIR/data/questionnaires"

mapfile -t PATIENT_FILES < <(find "$PATIENTS_DIR" -maxdepth 2 -name "*_001.yaml" | sort)
mapfile -t QUESTIONNAIRES < <(
    PYTHONPATH="$ROOT_DIR" "$PYTHON" - <<'PY'
from agent.core.questionnaire_catalog import iter_questionnaire_definitions, questionnaire_is_runnable

for questionnaire_def in iter_questionnaire_definitions():
    if questionnaire_is_runnable(questionnaire_def):
        print(questionnaire_def["id"])
PY
)

if [[ ${#PATIENT_FILES[@]} -eq 0 ]]; then
    echo "No canonical patient files ending with _001 found in $PATIENTS_DIR" >&2
    exit 1
fi
if [[ ${#QUESTIONNAIRES[@]} -eq 0 ]]; then
    echo "No runnable questionnaire files found in $QUESTIONNAIRES_DIR" >&2
    exit 1
fi

# Extract IDs (filename stem, no extension)
PATIENTS=()
for f in "${PATIENT_FILES[@]}"; do
    PATIENTS+=("$(basename "$f" .yaml)")
done

N_PATIENTS=${#PATIENTS[@]}
N_QUESTIONNAIRES=${#QUESTIONNAIRES[@]}
TOTAL=$(( N_PATIENTS * N_QUESTIONNAIRES ))
DONE=0
FAILED=0
SKIPPED=0

echo "========================================================"
echo "  LLMPatients Questionnaire Batch Runner"
echo "========================================================"
echo "  Patients       : $N_PATIENTS"
echo "  Questionnaires : $N_QUESTIONNAIRES"
echo "  Total pairs    : $TOTAL"
[[ -n "$FORCE_FLAG" ]] && echo "  Mode           : FORCE (overwrite existing results)"
echo "========================================================"
echo ""

START_TIME=$(date +%s)

for PATIENT in "${PATIENTS[@]}"; do
    for Q in "${QUESTIONNAIRES[@]}"; do
        DONE=$(( DONE + 1 ))
        RESULT_FILE="$ROOT_DIR/data/questionnaire_results/$PATIENT/$Q.json"

        # Without --force, skip pairs that already have a result file
        if [[ -z "$FORCE_FLAG" && -f "$RESULT_FILE" ]]; then
            echo "[$DONE/$TOTAL] SKIP  $PATIENT / $Q (result exists)"
            SKIPPED=$(( SKIPPED + 1 ))
            continue
        fi

        echo "[$DONE/$TOTAL] RUN   $PATIENT / $Q"

        if PYTHONPATH="$ROOT_DIR" "$PYTHON" "$RUNNER" \
                --patient "$PATIENT" \
                --questionnaire "$Q" \
                $FORCE_FLAG; then
            echo "[$DONE/$TOTAL] OK    $PATIENT / $Q"
        else
            echo "[$DONE/$TOTAL] FAIL  $PATIENT / $Q" >&2
            FAILED=$(( FAILED + 1 ))
        fi

        echo ""
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
SUCCEEDED=$(( TOTAL - FAILED - SKIPPED ))

echo "========================================================"
echo "  Batch complete"
echo "  Elapsed  : ${ELAPSED}s"
echo "  Succeeded: $SUCCEEDED"
echo "  Skipped  : $SKIPPED"
echo "  Failed   : $FAILED"
echo "========================================================"

[[ $FAILED -eq 0 ]]   # exit 0 if no failures, 1 otherwise
