import json
import subprocess
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

CHAOS_LIBRARY_DIR = Path("chaos-library")
EXPERIMENT_MANIFEST = Path("experiments.json")
CHAOS_LOG_FILE = Path("chaos_log.json")


def run_kubectl(args: list[str], dry_run: bool = False):
    """Executes a kubectl command, handling errors and dry-run mode."""
    command = ["kubectl"] + args
    print(f"  > {' '.join(command)}")
    if dry_run:
        return
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  🚨 Error executing command: {e}")
        print(f"  🚨 STDERR: {e.stderr}")
        raise

def main(dry_run: bool):
    """Main function to orchestrate the chaos experiments."""
    print(f"CHAOS ORCHESTRATOR STARTED{' (DRY RUN)' if dry_run else ''}")
    print("-" * 40)

    if not EXPERIMENT_MANIFEST.exists():
        print(f"🚨 Error: Manifest file not found at {EXPERIMENT_MANIFEST}")
        return

    with open(EXPERIMENT_MANIFEST, "r") as f:
        manifest = json.load(f)

    logs = []
    if CHAOS_LOG_FILE.exists():
        with open(CHAOS_LOG_FILE, "r") as f:
            logs = json.load(f)
        print(f"Loaded {len(logs)} existing log entries from {CHAOS_LOG_FILE}")

    for i, experiment in enumerate(manifest["experiments"]):
        exp_id = experiment["id"]
        description = experiment["description"]
        duration = experiment["duration_seconds"]
        yaml_files = [CHAOS_LIBRARY_DIR / f for f in experiment["files"]]

        print(f"\n▶️ Running Experiment {i+1}/{len(manifest['experiments'])}: {exp_id}")
        print(f"  Description: {description}")
        print(f"  Duration: {duration} seconds")

        start_time, end_time = None, None
        try:
            start_time = datetime.now(timezone.utc)
            print("  Applying chaos resources...")
            for yaml_file in yaml_files:
                run_kubectl(["apply", "-f", str(yaml_file)], dry_run)

            print(f"  Chaos active. Waiting for {duration} seconds...")
            if not dry_run:
                time.sleep(duration)

        except Exception as e:
            print(f"🚨 An unexpected error occurred during experiment {exp_id}: {e}")
        finally:
            end_time = datetime.now(timezone.utc)
            print("  Cleaning up chaos resources...")
            for yaml_file in yaml_files:
                try:
                    run_kubectl(["delete", "-f", str(yaml_file)], dry_run)
                except Exception as e:
                    print(f"  🚨 Warning: Cleanup for {yaml_file.name} failed. Manual check may be needed.")

            if dry_run:
                print("  [Dry Run] Skipping log entry.")
                continue

            log_entry = {
                "experiment_id": exp_id,
                "description": description,
                "tier": experiment["tier"],
                "chaos_files": [f.name for f in yaml_files],
                "start_time_utc": start_time.isoformat() if start_time else "N/A",
                "end_time_utc": end_time.isoformat() if end_time else "N/A",
                "duration_seconds": duration,
                "status": "COMPLETED" if start_time else "FAILED"
            }
            logs.append(log_entry)

            # Write back to the log file after each experiment (transactional)
            with open(CHAOS_LOG_FILE, "w") as f:
                json.dump(logs, f, indent=2)
            print(f"  ✅ Log entry saved for {exp_id}.")

    print("-" * 40)
    print("CHAOS ORCHESTRATOR FINISHED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chaos Engineering Orchestrator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the execution without applying any kubectl commands."
    )
    args = parser.parse_args()
    main(args.dry_run)
