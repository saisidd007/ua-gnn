#!/usr/bin/env python
"""Simple training completion monitor and experiment auto-runner"""
import subprocess
import os
import time
from pathlib import Path

def check_training_complete(log_file="results/train_50.log"):
    """Check if training has completed"""
    if not os.path.exists(log_file):
        return False
    
    try:
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()
            if not lines:
                return False
            last_line = lines[-1].strip()
            return any(x in last_line for x in ["Training completed", "Early stopping", "[SUCCESS]", "[DONE]"])
    except:
        return False

def run_exp(cmd, desc):
    """Run experiment"""
    print(f"\n{'='*60}\n[EXP] {desc}\n{'='*60}")
    try:
        subprocess.run(cmd, timeout=3600)
        print(f"✓ {desc} done")
        return True
    except Exception as e:
        print(f"✗ {desc} failed: {e}")
        return False

def main():
    os.chdir(Path(__file__).parent)
    py = r'C:\venv_traffic_flow\Scripts\python.exe'
    
    print("[START] Monitoring training...")
    start = time.time()
    check_count = 0
    
    # Monitor for up to 48 hours
    while time.time() - start < 172800:
        check_count += 1
        if check_training_complete():
            elapsed = (time.time() - start) / 3600
            print(f"\n[DONE] Training completed in {elapsed:.1f} hours")
            break
        
        elapsed = (time.time() - start) / 3600
        print(f"[CHECK {check_count}] Elapsed: {elapsed:.1f}h - Training still running...", end='\r')
        time.sleep(60)  # Check every minute
    else:
        print(f"\n[TIMEOUT] Max wait exceeded")
        return
    
    # Run experiments
    exps = [
        ([py, "notebooks/analysis_50epoch.py"], "Per-horizon metrics"),
        ([py, "scripts/sensor_dropout_experiment.py"], "Sensor dropout"),
        ([py, "scripts/multiple_runs_analysis.py"], "Multiple runs"),
        ([py, "scripts/calibration_analysis.py"], "Calibration"),
    ]
    
    results = []
    for cmd, desc in exps:
        results.append((desc, run_exp(cmd, desc)))
    
    print(f"\n{'='*60}\n[SUMMARY]\n{'='*60}")
    for desc, ok in results:
        print(f"{'✓' if ok else '✗'} {desc}")
    print(f"\nPassed: {sum(1 for _, ok in results if ok)}/{len(results)}")

if __name__ == "__main__":
    main()
