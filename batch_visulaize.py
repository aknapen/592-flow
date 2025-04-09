import subprocess

# Constants (modify these as needed)
base_command = "python3 visualizer_rllib.py"
base_path = "~/ray_results/fleet_control/"
experiment_id = "PPO_FleetControlEnv-v0_ebb2fe38_2025-04-09_13-41-51lb37qlpb"
fixed_arg = "100"
render_mode = "--render_mode='no_render'"
output_dir = "results/15v_5g/"
discount = "16times3"

# List of `c` values you want to run
# c_values = list(range(20, 51, 20))
c_values = [20, 40, 60, 80, 100]

# Loop through each `c` value and run the command
for c in c_values:
    output_file = f"{output_dir}rl_{discount}_45h_5r_{fixed_arg}s_{c}c_vis.txt"
    cmd = f"{base_command} {base_path}{experiment_id} {c} {render_mode} > {output_file}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, executable='/bin/bash')
