import os
import argparse


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_run', help='The total number of experiments.', required=True)
parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
parser.add_argument('-g', '--gpu_id', help='The gpu index.', default=0, required=False)
args = parser.parse_args()

for i in range(int(args.num_run)):
    print(f"Start carrying out experiment {i+1}/{args.num_run}...")
    exec_str = f"CUDA_VISIBLE_DEVICES={args.gpu_id} python run.py -f {args.yaml_file} --multi_run_mode"
    os.system(exec_str)
    print("\n\n")
