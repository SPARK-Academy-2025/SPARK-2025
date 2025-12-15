# main.py
import argparse
import os
import torch
from tools.inference_utils_old import run_inference
from model import ModalitySelectiveUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModalitySelectiveUNet().to(device)
    model.load_state_dict(torch.load(
        "checkpoints/best_final_model.pth", map_location=device))
    model.eval()

    run_inference(model, args.input, args.output, device)


if __name__ == "__main__":
    main()
