import os
import shutil
import random

MODALITIES = ["t2f", "t1n", "t1c", "t2w"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_modality_files(patient_id, src_folder, dst_folder):
    for modality in MODALITIES:
        fname = f"{patient_id}-{modality}.nii.gz"
        src = os.path.join(src_folder, fname)
        dst = os.path.join(dst_folder, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Missing modality: {src}")

def copy_pretub_file(patient_id, pretub_folder, dst_images_folder):
    pretub_src = os.path.join(pretub_folder, f"{patient_id}-seg_perturbed.nii.gz")
    pretub_dst = os.path.join(dst_images_folder, f"{patient_id}-seg_perturbed.nii.gz")
    if os.path.exists(pretub_src):
        shutil.copy(pretub_src, pretub_dst)
    else:
        print(f"Missing pretub: {pretub_src}")

def copy_seg_file(patient_id, patient_folder, dst_masks_folder):
    seg_src = os.path.join(patient_folder, f"{patient_id}-seg.nii.gz")
    seg_dst = os.path.join(dst_masks_folder, f"{patient_id}-seg.nii.gz")
    if os.path.exists(seg_src):
        shutil.copy(seg_src, seg_dst)
    else:
        print(f"Missing seg: {seg_src}")

def copy_prediction_placeholder(patient_id, pred_folder, dst_preds_folder):
    ensure_dir(dst_preds_folder)
    pred_src = os.path.join(pred_folder, f"{patient_id}_perturbed.nii.gz")
    pred_dst = os.path.join(dst_preds_folder, f"{patient_id}_perturbed.nii.gz")
    if os.path.exists(pred_src):
        shutil.copy(pred_src, pred_dst)
    else:
        print(f"Missing prediction: {pred_src}")

def create_patient_split(patient_id, src_root, preds_dir, pretub_dir, dst_root, split_name, config=None):
    src_patient_folder = os.path.join(src_root, patient_id)
    dst_patient_root = os.path.join(dst_root, split_name, patient_id)
    images_dst = os.path.join(dst_patient_root, "images")
    masks_dst = os.path.join(dst_patient_root, "masks")
    preds_dst = os.path.join(dst_patient_root, "predictions")

    ensure_dir(images_dst)
    copy_modality_files(patient_id, src_patient_folder, images_dst)

    if (config and config.get('has_pretub', True)):
        copy_pretub_file(patient_id, pretub_dir, images_dst)

    ensure_dir(masks_dst)
    copy_seg_file(patient_id, src_patient_folder, masks_dst)

    if (config and config.get('has_preds', True)):
        ensure_dir(images_dst)
        copy_prediction_placeholder(patient_id, preds_dir, images_dst)

def split_dataset(patient_ids, train_ratio, val_ratio, test_ratio):
    random.shuffle(patient_ids)
    total = len(patient_ids)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    return {
        'train': patient_ids[:train_end],
        'val': patient_ids[train_end:val_end],
        'test': patient_ids[val_end:]
    }

def main(config):
    dataset_root = config.get('dataset_root')
    pretub_dir = config.get('pretub_dir')
    preds_dir = config.get('preds_dir')
    output_root = config.get('output_root')
    split_names = config.get('split_names')
    train_ratio = config.get('train_ratio')
    val_ratio = config.get('val_ratio')
    test_ratio = config.get('test_ratio')

    # Get list of patient folders (excluding 'pertubation' and 'prediction')
    patient_ids = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d != "pertubation"
        if os.path.isdir(os.path.join(dataset_root, d)) and d != "prediction"
    ])
    splits = split_dataset(patient_ids, train_ratio, val_ratio, test_ratio)

    for split_key, patient_list in splits.items():
        split_name = split_names[split_key]
        for patient_id in patient_list:
            create_patient_split(
                patient_id=patient_id,
                src_root=dataset_root,
                preds_dir=preds_dir,
                pretub_dir=pretub_dir,
                dst_root=output_root,
                split_name=split_name,
                config=config
            )

    # Print summary
    print("\n Split complete:")
    for split_key, patient_list in splits.items():
        print(f"{split_names[split_key]}: {len(patient_list)} patients")

def configs():
    dataset_root = "data/raw/ssa"
    output_root = "data/split/ssa"

    return {
        "dataset_root" : dataset_root,
        "pretub_dir" : os.path.join(dataset_root, "pertubation"),
        "preds_dir" : os.path.join(dataset_root, "prediction"),
        "output_root" : output_root,
        "has_pretub" : True,
        "has_preds" : False,
        "split_names" : {
            'train': 'Training',
            'val': 'Validation',
            'test': 'Testing',
        },
        "train_ratio" : 0.7,
        "val_ratio" : 0.25,
        "test_ratio" : 0.05,
    }

if __name__ == "__main__":
    random.seed(42)
    main(config=configs())
