"""Run the whole pipeline in order:  python run_all.py
Run one step:                         python -m pipeline.step3_quality_control
Start from a later step:              python run_all.py 4      (steps 4, 5, 6, 7)
Optional steps (not run here):        python -m pipeline.step8_gru_optional
                                      python -m pipeline.step9_drought_optional
"""
import importlib
import sys
import time

STEPS = [
    "step1_observations",       # station truth (SYNOP)            ~20 min the first time
    "step2_forecasts",          # IFS / AIFS / GFS old forecasts    ~25 min the first time
    "step3_quality_control",
    "step4_features",
    "step5_train_evaluate",     # walk-forward training            ~15-30 min
    "step6_report",
    "step7_explain",
]

if __name__ == "__main__":
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for number, name in enumerate(STEPS, start=1):
        if number < first:
            continue
        print(f"\n===== STEP {number}: {name} =====")
        start = time.time()
        importlib.import_module(f"pipeline.{name}").main()
        print(f"===== step {number} finished in {time.time() - start:.0f} s =====")
