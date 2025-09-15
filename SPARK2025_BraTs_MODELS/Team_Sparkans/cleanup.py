import os
import re
import argparse

parser = argparse.ArgumentParser("For renaming final prediction files and cleaning up other stuff")
parser.add_argument('-i', type = str, required = True)

args = parser.parse_args()

files = os.listdir(args.i)

for file in files:
    if not "dataset.json" in file: 
        idx = re.search("BraTS_(.*).nii.gz", file).group(1)
        os.rename(f"{args.i}/{file}", f"{args.i}/BraTS-SSA-{idx}-000.nii.gz")

    else:
        os.remove(f"{args.i}/{file}")