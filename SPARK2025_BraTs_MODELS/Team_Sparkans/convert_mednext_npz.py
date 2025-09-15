import os
import numpy as np
import pickle
from tqdm import tqdm

def pad_back_to_original_shape(cropped, bbox, target_shape=(4, 155, 240, 240)):
    padded = np.zeros(target_shape, dtype=np.float32)
    (z_start, z_end), (y_start, y_end), (x_start, x_end) = bbox
    padded[:, z_start:z_end, y_start:y_end, x_start:x_end] = cropped
    return padded

def load_bbox(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['bbox_used_for_cropping']

def process_all_predictions(npz_dir, bbox_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for file in tqdm(os.listdir(npz_dir), desc="Processing predictions"):
        if not file.endswith('.npz'):
            continue

        pid = os.path.splitext(file)[0]
        npz_path = os.path.join(npz_dir, file)
        bbox_path = os.path.join(bbox_dir, pid + '.pkl')
        out_path = os.path.join(out_dir, pid + '.npz')

        if not os.path.exists(bbox_path):
            print(f"[!] Missing bbox for {pid}")
            continue

        npz_file = np.load(npz_path)
        first_key = list(npz_file.keys())[0]
        probs = npz_file[first_key]

        bbox = load_bbox(bbox_path)
        restored = pad_back_to_original_shape(probs, bbox)
        np.savez_compressed(out_path, probabilities=restored)

def main():
    # -------------------
    # CONFIGURE HERE
    # -------------------
    import argparse

    parser = argparse.ArgumentParser("For converting mednext files")
    parser.add_argument('-in_mednext', type = str, required = True,
                        help = "Path to folder containing MedNeXT *.npz files")
    parser.add_argument('-in_pkl', type = str, required = True, 
                        help = 'Path to folder containing nnunet *.pkl files')
    parser.add_argument('-o', type = str, required =  True, 
                        help = 'Path to output mednext ensembled files')

    args = parser.parse_args()

    if not os.path.isdir(args.o):
        os.makedirs(args.o, exist_ok=True)

    process_all_predictions(args.in_mednext, args.in_pkl, args.o)

if __name__ == "__main__":
    main()
