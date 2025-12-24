import argparse
from tools.nnunet_function import nnunet_infer


parser = argparse.ArgumentParser(description="Run nnUNet inference and merge masks")

parser.add_argument('-i', '--input', required=True, help="Path to input data")
parser.add_argument('-o', '--output', required=True, help="Path to output directory")
args = parser.parse_args()
input_path = args.input
output_path = args.output
tools_path = "tools"
nnUNet_path = f"tools/nnUNet"
mid_output_path = f"{nnUNet_path}/midoutput"
output_50_path = f"{mid_output_path}/50"
checkpoint_50_path = "checkpoint/50"
nnunet_infer(dataset_path=input_path, output_path=output_50_path, nnunet_path=nnUNet_path, checkpoint_dir=checkpoint_50_path)