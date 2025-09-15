import os
import re
import argparse
import shutil

parser = argparse.ArgumentParser("For preparing the input files in the correct format.")
parser.add_argument('-i', type = str, required = True)
parser.add_argument('-o', type = str, required = True)

args = parser.parse_args()

if not os.path.isdir(args.o):
    os.makedirs(args.o, exist_ok=True)

folders = os.listdir(args.i)

for folder in folders:
    top_dir = args.i + "/" + folder
    files = os.listdir(top_dir)

    idx = re.search("BraTS-SSA-(.*)-000", folder).group(1)

    for file in files:
        print(f"Now copying file: {file} to /temp")

        if "t1c" in file:
            shutil.copy(f"{args.i}/{folder}/{file}", f"{args.o}/BraTS_{idx}_0000.nii.gz")

        elif "t1n" in file:
            shutil.copy(f"{args.i}/{folder}/{file}", f"{args.o}/BraTS_{idx}_0001.nii.gz")

        elif "t2f" in file:
            shutil.copy(f"{args.i}/{folder}/{file}", f"{args.o}/BraTS_{idx}_0002.nii.gz")

        elif "t2w" in file:
            shutil.copy(f"{args.i}/{folder}/{file}", f"{args.o}/BraTS_{idx}_0003.nii.gz")