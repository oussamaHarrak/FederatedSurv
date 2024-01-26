from importlib.metadata import distribution
import subprocess
import os
import re

def update_envoy_config(base_dir, envoy_rank, num_envoys):
    envoy_config_path = os.path.join(base_dir, f"envoy_{envoy_rank}/envoy_config.yaml")

    # Read the contents of the envoy_config.yaml file
    with open(envoy_config_path, 'r') as file:
        config_content = file.read()

    # Find and replace the rank of the envoy
    pattern = r'rank_worldsize:\s*\d+,\s*\d+'
    new_config_content = re.sub(pattern, f'rank_worldsize: {envoy_rank}, {num_envoys}', config_content)

    # Write the updated content back to the file
    with open(envoy_config_path, 'w') as file:
        file.write(new_config_content)

def update_descriptor(base_dir,  num_envoys , distribution):
    
    # Get the last part of the base_dir path
    dir_name = os.path.basename(os.path.normpath(base_dir))

    # Remove any file extension from the directory name
    dir_name = os.path.splitext(dir_name)[0]

    # Find the position of the first underscore in the directory name
    underscore_index = dir_name.find('_')

    # Get the substring after the first underscore (excluding the underscore)
    if underscore_index != -1:
        dir_name = dir_name[underscore_index + 1:]

    # Convert the first character to lowercase and concatenate the rest of the directory name
    file_name = dir_name.lower() + "_shard_descriptor"
    for envoy_rank in range(1, num_envoys + 1):
        descriptor_path = os.path.join(base_dir, f"envoy_{envoy_rank}/{file_name}.py")

        # Read the contents of the shared_descriptor.yaml file
        with open(descriptor_path, 'r') as file:
            content = file.read()
        # Check if the line is already present in the file
        parenthese_index = distribution.find('(')
        if underscore_index != -1:
            distribution_name = distribution[:parenthese_index]
        if f"from openfl.utilities.data_splitters.numpy import {distribution_name}" not in content:
            # If the line is not present, add it after the line "from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor"
            content = content.replace("from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor",
                                      f"from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor\n\nfrom openfl.utilities.data_splitters.numpy import {distribution_name}")

            # Write the updated content back to the file
            with open(descriptor_path, 'w') as file:
                file.write(content)

            print("Line added.")
        else:
            print("Line already present.")
        # Define the pattern with a group to capture the existing splitter assignment
        pattern = r'(splitter\s*=\s*)(.*)'

        # Replace the pattern with the new distribution name
        new_content, replacements = re.subn(pattern, rf'\1{distribution}', content)

        if replacements > 0:
            # Write the updated content back to the file
            with open(descriptor_path, 'w') as file:
                file.write(new_content)

            print(f"Line replaced: {pattern} -> splitter = {distribution}")
        else:
            print("Pattern not found in the file.")

    descriptor_path = os.path.join(base_dir, f"workspace/{file_name}.py")
    with open(descriptor_path, 'r') as file:
            content = file.read()
    # Check if the line is already present in the file
    parenthese_index = distribution.find('(')
    if underscore_index != -1:
        distribution_name = distribution[:parenthese_index]

    if f"from openfl.utilities.data_splitters.numpy import {distribution_name}" not in content:
        # If the line is not present, add it after the line "from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor"
        content = content.replace("from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor",
                                      f"from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor\n\nfrom openfl.utilities.data_splitters.numpy import {distribution_name}")

        # Write the updated content back to the file
        with open(descriptor_path, 'w') as file:
            file.write(content)

        print("Line added.")
    else:
        print("Line already present.")
    # Define the pattern with a group to capture the existing splitter assignment
    pattern = r'(splitter\s*=\s*)(.*)'

    # Replace the pattern with the new distribution name
    new_content, replacements = re.subn(pattern, rf'\1{distribution}', content)

    if replacements > 0:
        # Write the updated content back to the file
        with open(descriptor_path, 'w') as file:
            file.write(new_content)

        print(f"Line replaced: {pattern} -> splitter = {distribution}")
    else:
        print("Pattern not found in the file.")

def generate_commands( base_dir , num_envoys, distribution ):
    commands = []
    director_cmd = f"ssh medium-01 conda activate myenv && cd {os.path.join(base_dir, 'director')} && fx director start --disable-tls -nn False -c director.yaml"
    commands.append(director_cmd)

    # Replace "myenv" with the name of your Anaconda environment
    for i in range(1, num_envoys + 1):
        envoy_cmd = f"ssh medium-0{i+1} conda activate myenv && cd {os.path.join(base_dir, f'envoy_{i}')} && fx envoy start -n envoy_{i} --disable-tls --envoy-config-path envoy_config.yaml -dh localhost -dp 50054"
        update_envoy_config(base_dir, i, num_envoys)
        commands.append(envoy_cmd)

    if distribution != "uniform": 
        update_descriptor(base_dir, num_envoys , distribution)

    script_name = os.path.basename(os.path.normpath(base_dir))
    workspace_cmd = f"ssh medium-0{i} conda activate myenv && cd {os.path.join(base_dir, 'workspace')} && python {script_name}.py"
    commands.append(workspace_cmd)

    return commands

def extract_metrics_from_output(output):
    # Search for the 4_adaboost_validate section in the output
    section_match = re.search(r'4_adaboost_validate(.*?)(?=4_adaboost_validate|$)', output, re.DOTALL)

    if section_match:
        section_output = section_match.group(1)

        # Search for the Concordance-index and Integrated-Brier-Score metrics in the 4_adaboost_validate section
        concordance_index_match = re.search(r'Concordance-index:\s*([\d.]+)', section_output)
        integrated_brier_score_match = re.search(r'Integrated-Brier-Score:\s*([\d.]+)', section_output)
        
        # Get the values of the metrics if found
        concordance_index = float(concordance_index_match.group(1)) if concordance_index_match else None
        integrated_brier_score = float(integrated_brier_score_match.group(1)) if integrated_brier_score_match else None

        return concordance_index, integrated_brier_score

    return None, None

# Function to create and run batch files

def run_commands_in_separate_prompts(commands):
    # Create a list to store the processes
    processes = []

    # Start each command in a separate batch file and store the process in the 'processes' list
    for i, cmd in enumerate(commands):
        batch_file = f"command_{i}.bat"
        with open(batch_file, "w") as f:
            f.write(cmd)

        try:
            process = subprocess.Popen(["start", "cmd", "/K", batch_file], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(process)

        except subprocess.CalledProcessError as e:
            print("Error:", e)

    # Wait for the 'director' process to finish
    processes[0].wait()
    # Extract metrics for the 'director' process (command_0.bat)
    output = processes[0].communicate()[0].decode("utf-8")

    concordance_index, integrated_brier_score = extract_metrics_from_output(output)

    if concordance_index is not None and integrated_brier_score is not None:
        # Write the metrics to a text file
        with open("metrics_command_0.txt", "w") as file:
            file.write(f"Concordance-index: {concordance_index}\n")
            file.write(f"Integrated-Brier-Score: {integrated_brier_score}\n")

        print("Metrics for command_0 (director) saved to metrics_command_0.txt")
    else:
        print("Metrics for command_0 (director) not found in the output.")

    # Now, process the outputs of other commands
    for i, process in enumerate(processes[1:], start=1):
        output = process.communicate()[0].decode("utf-8")
        # For other commands (envoy and workspace), skip processing the output
        print(f"Command_{i} executed. Output not processed.")

"""
def run_commands_in_separate_prompts(commands):
    import subprocess

    # Create a list to store the processes
    processes = []

    # Start each command in a separate batch file and store the process in the 'processes' list
    for i, cmd in enumerate(commands):
        batch_file = f"command_{i}.bat"
        log_file = f"log_command_{i}.txt"  # Log file to capture the output
        with open(batch_file, "w") as f:
            f.write(f"{cmd} >> {log_file} 2>&1")  # Redirect both stdout and stderr to the log file

        try:
            process = subprocess.Popen(["start", "cmd", "/C", batch_file], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(process)

        except subprocess.CalledProcessError as e:
            print("Error:", e)

    # Wait for the 'director' process to finish
    processes[0].wait()

    # Extract metrics for the 'director' process (command_0.bat)
    with open(f"log_command_0.txt", "r") as log_file:
        output = log_file.read()

    concordance_index, integrated_brier_score = extract_metrics_from_output(output)

    if concordance_index is not None and integrated_brier_score is not None:
        # Write the metrics to a text file
        with open("metrics_command_0.txt", "w") as file:
            file.write(f"Concordance-index: {concordance_index}\n")
            file.write(f"Integrated-Brier-Score: {integrated_brier_score}\n")

        print("Metrics for command_0 (director) saved to metrics_command_0.txt")
    else:
        print("Metrics for command_0 (director) not found in the output.")

    # Now, process the outputs of other commands
    for i, process in enumerate(processes[1:], start=1):
        process.wait()
        # For other commands (envoy and workspace), skip processing the output
        print(f"Command_{i} executed. Output not processed.")
"""
# Get the number of envoys from user input
num_envoys = int(input("Enter the number of envoys: "))

# Get the base directory from user input
base_dir = input("Enter the base directory: ")


distribution = input("Enter the distribution : ")
    
# Generate commands based on the number of envoys and the base directory
commands = generate_commands( base_dir , num_envoys , distribution )

# Run the commands in separate Anaconda prompts
run_commands_in_separate_prompts(commands)
"""
def run_commands_in_separate_prompts(commands):
    # Create a list to store the processes
    processes = []

    # Start each command in a separate batch file and store the process in the >
    for i, cmd in enumerate(commands):
        batch_file = f"command_{i}.bat"
        with open(batch_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)

        try:
            process = subprocess.Popen(
                ["bash", batch_file],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(process)
            print(f"Command_{i} started. Log will be saved to 'log_command_{i}.>
        except subprocess.CalledProcessError as e:
            print("Error:", e)

    for i, process in enumerate(processes[1:], start=1):
        process.wait()
        print(f"Command_{i} finished. Log is saved to 'log_command_{i}.txt'.")
"""