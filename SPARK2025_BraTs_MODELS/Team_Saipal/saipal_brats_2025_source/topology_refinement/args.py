import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Path to checkpoint')

    return parser.parse_args()

