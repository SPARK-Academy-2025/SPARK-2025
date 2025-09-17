

import argparse
from tools.nnunet_function import nnunet_infer, merge_masks


parser = argparse.ArgumentParser(description="Run nnUNet inference and merge masks")

parser.add_argument('-i', '--input', required=True, help="Path to input data")
parser.add_argument('-o', '--output', required=True, help="Path to output directory")
args = parser.parse_args()
input_path = args.input
output_path = args.output
tools_path = "tools"
nnUNet_path = f"tools/nnUNet"
mid_output_path = f"{nnUNet_path}/midoutput"
output_200_path = f"{mid_output_path}/200"
output_3500_path = f"{mid_output_path}/3500"
checkpoint_200_path = "checkpoint/200"
checkpoint_3500_path = "checkpoint/3500"

nnunet_infer(dataset_path=input_path, output_path=output_200_path, nnunet_path=nnUNet_path, checkpoint_dir=checkpoint_200_path)
nnunet_infer(dataset_path=input_path, output_path=output_3500_path, nnunet_path=nnUNet_path, checkpoint_dir=checkpoint_3500_path)
merge_masks(whole_tumor_dir=output_200_path, other_dir=output_3500_path, output_dir=output_path)