# Evaluations

This folder organizes evaluation code by scenario. Each scenario has:
- a dedicated script to run the evaluation
- a README describing purpose, rationale, and how to interpret results

Current structure (default scenario shown):

- evaluations/
  - scenarios/
    - pair_source_verification/  # default pairwise source verification evaluation
      - eval_pair_model.py
      - README.md

To add a new scenario, create a new folder under `evaluations/scenarios/` with its
own script and README. Keep the job launchers in `jobs/` and point them at the
scenario script.
