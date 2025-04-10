import subprocess
import csv
import os
import re

# Constants (modify these as needed)
base_command = "python3 visualizer_rllib.py"
base_path = os.path.expanduser("~/ray_results/fleet_control/")
experiment_id = "PPO_FleetControlEnv-v0_aa55c97a_2025-04-09_14-15-49sb0ae23e"
fixed_arg = "10000"
render_mode = "--render_mode='no_render'"
output_dir = "results/15v_5g/"
discount = "32times3"
csv_output_path = f"{output_dir}summary_{discount}.csv"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List of `c` values you want to run

c_values = list(range(120, 10001, 20))

# c_values = [20, 40, 60, 80, 100]

# Open the CSV file for writing
with open(csv_output_path, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(['c', 'value'])  # Write headers

    # Loop through each `c` value and run the command
    for c in c_values:
        output_file = f"{output_dir}rl_{discount}_45h_5r_{fixed_arg}s_{c}c_vis.txt"
        cmd = f"{base_command} {base_path}{experiment_id} {c} {render_mode}"
        print(f"Running: {c, cmd}")

        # Run command and capture output
        result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)
        output = result.stdout

        # Save output to file
        with open(output_file, 'w') as f:
            f.write(output)

         # Extract value from first line
        first_line = output.splitlines()[0] if output else ""
        match = re.search(r"Return:\s*([\d.]+)", first_line)
        return_value = float(match.group(1)) if match else 'NaN'

        # Write to CSV
        writer.writerow([c, return_value])