import argparse
import subprocess
import time
from datetime import datetime


def run_command(cmd):
    """Runs a shell command and splits the output into rows/columns."""
    try:
        # stderr=subprocess.DEVNULL hides error spam if a pod is terminating
        output = subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        )
        return [line.split() for line in output.strip().split("\n") if line]
    except subprocess.CalledProcessError:
        return []


def parse_cpu(val):
    """Converts CPU strings (e.g., '150m', '1') into raw millicores (float)."""
    val = val.strip()
    if val.endswith("m"):
        return float(val[:-1])
    try:
        return float(val) * 1000  # Convert whole cores to millicores
    except ValueError:
        return 0.0


def parse_mem(val):
    """Converts Memory strings (e.g., '256Mi', '1Gi') into raw MiB (float)."""
    val = val.strip()
    if val.endswith("Mi"):
        return float(val[:-2])
    if val.endswith("Gi"):
        return float(val[:-2]) * 1024
    if val.endswith("Ki"):
        return float(val[:-2]) / 1024
    try:
        return float(val)
    except ValueError:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Log K8s resources to CSV format")
    parser.add_argument("--interval", type=int, default=2, help="Seconds between polls")
    args = parser.parse_args()

    # Print the CSV Header
    print("timestamp,entity_type,entity_name,cpu_millicores,memory_mib")

    try:
        while True:
            # Get current time for the X-axis of your future plot
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 1. Gather Node Metrics
            # Format: NAME  CPU(cores)  CPU%  MEMORY(bytes)  MEMORY%
            nodes = run_command("kubectl top nodes --no-headers")
            for cols in nodes:
                if len(cols) >= 4:
                    name = cols[0]
                    cpu = parse_cpu(cols[1])
                    mem = parse_mem(cols[3])
                    print(f"{now},node,{name},{cpu},{mem}")

            # 2. Gather Pod Metrics
            # Format: NAME  CPU(cores)  MEMORY(bytes)
            pods = run_command("kubectl top pods -n fedroom --no-headers")
            for cols in pods:
                if len(cols) >= 3:
                    name = cols[0]
                    cpu = parse_cpu(cols[1])
                    mem = parse_mem(cols[2])
                    print(f"{now},pod,{name},{cpu},{mem}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nLogging stopped.")


if __name__ == "__main__":
    main()
