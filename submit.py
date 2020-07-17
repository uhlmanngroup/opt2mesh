#!/usr/bin/env python
import os
import sys
import yaml
import itertools
from collections import ChainMap

__doc__ = """

    Job submitter on a yaml request specified like this.

    Example:

    ```yaml
    batch_job_name: crf
    goal: This batch of jobs is dedicated to explore different values \n
          of parameters for CRF on two different input files.\n\n
          This would results in 32 total jobs.
    python_exec: /homes/jerphanion/.miniconda3/envs/ec/bin/python
    file: /homes/jerphanion/mesh-processing-pipeline/src/python/crf.py
    memory: 20000
    cpus: 4
    input_files:
      - /homes/jerphanion/mesh-processing-pipeline/data/original/MNS_M897_115.tif
      - /homes/jerphanion/mesh-processing-pipeline/data/original/MNS_M851_12.tif
    output_folder: /homes/jerphanion/mesh-processing-pipeline/out/crf
    options:
      - max_iter: [2, 3]
      - std_pos: [0.2, 0.5]
      - weight_pos: [0.7, 1.0]
      - std_bilat: [3.0, 5.0]
      - weight_bilat: [15.0]
    ```
"""


def bsub_command(python_exec, file, batch_job_name,
                 memory, cpus, input_file, output_folder, options):
    options_str = ""
    for k, v in options.items():
        options_str += f" --{k} {v} \ \n"

    command = "bsub \ \n"
    command += f" -J {batch_job_name} \ \n"
    command += f" -M {memory} \ \n"
    command += f' -R "rusage[mem={memory}]" \ \n'
    command += f' -n {cpus} \ \n'
    command += f'{python_exec} \ \n'
    command += f'{file} \ \n'
    command += f'{options_str}'
    command += f'{input_file} \ \n'
    command += f'{output_folder} \ \n'
    return command


if __name__ == "__main__":
    yaml_file = sys.argv[1]

    with open(yaml_file, 'r') as stream:
        try:
            raw = stream.read()
        except yaml.YAMLError as exc:
            print(exc)

    with open(yaml_file, 'r') as stream:
        try:
            command = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print(f"Content of {yaml_file}")
    print("```yaml")
    print(raw)
    print("```")

    python_exec = command["python_exec"]
    file = command["file"]
    batch_job_name = command["batch_job_name"]
    memory = command["memory"]
    cpus = command["cpus"]
    inputs_list = command["input_files"]
    output_folder = command["output_folder"]

    options_set = command["options"]

    # Merging dict together
    options_set = dict(ChainMap(*options_set))

    options_keys = options_set.keys()
    options_vals = list(options_set.values())

    bsub_commands = []

    # Creating commands for each files
    for input_file in inputs_list:

        # Submitting all the jobs possible on the cartesian product of options
        for values in itertools.product(*options_vals):
            options = dict(zip(options_keys, values))

            print(options)

            std_out_command = bsub_command(python_exec,
                                           file,
                                           batch_job_name,
                                           memory,
                                           cpus,
                                           input_file,
                                           output_folder,
                                           options)

            bsub_commands.append(std_out_command)
        else:
            std_out_command = bsub_command(python_exec,
                                           file,
                                           batch_job_name,
                                           memory,
                                           cpus,
                                           input_file,
                                           output_folder,
                                           dict())

            bsub_commands.append(std_out_command)

    n_jobs = len(bsub_commands)
    print(f"You are going to submit {n_jobs} jobs")

    if input("Confirm by entering 'y'").lower() == 'y':
        for i, command in enumerate(bsub_commands):
            print(f"Submitting job {i + 1} / {n_jobs}")
            std_out_command = os.popen(command.replace('\ \n', '')).read()
