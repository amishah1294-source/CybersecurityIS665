"""Run all steps in order:  python run_all.py"""
import subprocess, sys
for step in ["step1_check_data.py", "step2_explore_data.py", "step3_prepare_and_split.py", "step4_logistic_regression.py"]:
    print("\n=== " + step + " ===", flush=True)
    if subprocess.run([sys.executable, step]).returncode != 0:
        sys.exit(step + " failed")
print("\nDone!")
