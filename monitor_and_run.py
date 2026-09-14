#!/usr/bin/env python
"""Monitor training and auto-run experiments when complete"""
import subprocess
import time
import sys
import os
from pathlib import Path

def wait_for_training_complete(log_file="results/train_50.log", check_interval=10, max_wait=144000):
    """Wait for training to complete by monitoring log file"""
    start_time = time.time()
    last_size = 0
    
    print(f"[MONITOR] Waiting for training completion...")
    print(f"Log file: {log_file}")
    print(f"Check interval: {check_interval}s")
    
    while time.time() - start_time < max_wait:
        if os.path.exists(log_file):
            # Check file size for activity
            current_size = os.path.getsize(log_file)
            
            # Read last few lines to check for completion
            try:
                with open(log_file, 'r', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        
                        # Check for completion markers
                        if any(marker in last_line for marker in [
                            "Training completed",
                            "Early stopping triggered",
                            "[SUCCESS]",
                            "[DONE]"
                        ]):
                            print(f"\n[SUCCESS] Training completed!")
                            print(f"Last line: {last_line}")
                            return True
            except Exception as e:
                print(f"Warning: Could not read log: {e}")
        
        elapsed = (time.time() - start_time) / 3600
        size_change = "↑" if current_size > last_size else "=" if current_size == last_size else "?"
        print(f"[{time.strftime('%H:%M:%S')}] {size_change} Log size: {current_size:,} bytes | Elapsed: {elapsed:.1f}h", end='\r')
        last_size = current_size
        
        time.sleep(check_interval)
    
    print(f"\n[TIMEOUT] Max wait time ({max_wait}s = {max_wait/3600:.0f}h) exceeded")
    return False

def run_experiment(cmd, description):
    """Run an experiment script"""
    print(f"\n{'='*70}")
    print(f"[EXPERIMENT] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout per experiment
        if result.returncode == 0:
            print(f"✓ {description} completed")
            return True
        else:
            print(f"✗ {description} failed (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False

def main():
    os.chdir(Path(__file__).parent)
    python_exe = r"C:\venv_traffic_flow\Scripts\python.exe"
    
    if not os.path.exists(python_exe):
        print(f"ERROR: Python not found at {python_exe}")
        sys.exit(1)
    
    # Wait for training
    if not wait_for_training_complete():
        print("Training did not complete, exiting")
        sys.exit(1)
    
    # Run experiments
    experiments = [
        ([python_exe, "notebooks/analysis_50epoch.py"], "Per-horizon metrics"),
        ([python_exe, "scripts/sensor_dropout_experiment.py"], "Sensor dropout robustness"),
        ([python_exe, "scripts/multiple_runs_analysis.py"], "Multiple runs analysis"),
        ([python_exe, "scripts/calibration_analysis.py"], "Calibration analysis"),
    ]
    
    print("\n" + "="*70)
    print("[EXPERIMENTS] Running auto-experiments...")
    print("="*70)
    
    results = []
    for cmd, description in experiments:
        results.append((description, run_experiment(cmd, description)))
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY] Experiment Results:")
    print("="*70)
    for desc, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {desc}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} experiments passed")
    
    print("\n[COMPLETE] Pipeline finished!")
    print(f"Output files are in {os.path.join(os.getcwd(), 'results')}")

if __name__ == "__main__":
    main()
