# tools/inference_utils.py

import os
import numpy as np
import nibabel as nib
import torch
import scipy.ndimage as ndi
from pathlib import Path
from tqdm import tqdm
from monai.inferers import sliding_window_inference

# ------------------- CONFIG -------------------
# Modalities order must match training (bg, TC, ED, ET) -> channels: [t1ce, t2, flair]
MODS = ("-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz")

# Pre-crop target and sliding window
TARGET_SHAPE = (208, 192, 160)    # (Y, X, Z) after center fit
ROI_SIZE = (128, 128, 128)
SW_OVERLAP = 0.5
SW_BATCH = 1

# Cropping & normalization
BRAIN_MARGIN = 8        # voxels margin around brain bbox to avoid edge cuts
NORM_EPS = 1e-8

# TTA (flip ensembling). Choices below keep it fast on P100
USE_TTA = True
TTA_FLIPS = [None, (3,), (4,), (3, 4)]   # None, flip H, flip W, flip H+W
# If you can afford it, add depth flips too: (2,), (2,3), (2,4), (2,3,4)

# Post-processing to lift WT Dice by pruning small FPs
WT_MIN_CC_VOX = 50      # remove WT components smaller than this
# 0 = keep all above threshold; 1 = only largest, 2 = two largest, etc.
WT_KEEP_TOPK = 0
# ------------------------------------------------


def _safe_load_mod(case_dir: Path, suffix: str):
    """Load a NIfTI modality or raise a clear error."""
    g = list(case_dir.glob(f"*{suffix}"))
    if not g:
        raise FileNotFoundError(
            f"Missing modality {suffix} in {case_dir.name}")
    nii = nib.load(str(g[0]))
    return nii, nii.get_fdata().astype(np.float32)


def brain_mask_from_nonzero(t1c, t2w, t2f):
    """Binary brain from nonzeros + light morphology."""
    m = (t1c > 0) | (t2w > 0) | (t2f > 0)
    if not m.any():
        return m
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_erosion(m, iterations=1)
    return m


def bbox_from_mask(mask, margin=0, shape=None):
    """Return (slice_y, slice_x, slice_z) around nonzero mask with margin, clamped to shape."""
    ys, xs, zs = np.where(mask)
    if len(xs) == 0:
        return (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    z1, z2 = zs.min(), zs.max() + 1
    if margin and shape is not None:
        y1 = max(0, y1 - margin)
        y2 = min(shape[0], y2 + margin)
        x1 = max(0, x1 - margin)
        x2 = min(shape[1], x2 + margin)
        z1 = max(0, z1 - margin)
        z2 = min(shape[2], z2 + margin)
    return (slice(y1, y2), slice(x1, x2), slice(z1, z2))


def fit_to_shape(arr, tgt):
    """Center-crop/pad 3D array to target shape; return new_array, slices_used_on_orig, pads_used."""
    slices, pads = [], []
    for dim, t in zip(arr.shape, tgt):
        if dim > t:
            start = (dim - t) // 2
            sl = slice(start, start + t)
            slices.append(sl)
            pads.append((0, 0))
        else:
            sl = slice(None)
            before = (t - dim) // 2
            after = t - dim - before
            slices.append(sl)
            pads.append((before, after))
    arr2 = arr[tuple(slices)]
    arr2 = np.pad(arr2, pads, mode="constant")
    return arr2, slices, pads


def invert_fit(arr_fit, orig_shape, slices, pads):
    """Undo fit_to_shape back into the cropped (orig_shape) space."""
    (yb, ya), (xb, xa), (zb, za) = pads
    # Remove pad
    arr_unpadded = arr_fit[
        yb: None if ya == 0 else -ya,
        xb: None if xa == 0 else -xa,
        zb: None if za == 0 else -za
    ]
    out = np.zeros(orig_shape, dtype=arr_fit.dtype)
    out[slices[0], slices[1], slices[2]] = arr_unpadded
    return out


def zscore_per_mod_in_mask(vols_4d, mask):
    """
    Z-score per modality using voxels inside 'mask' only (ignores padded zeros/background).
    vols_4d: [C, Y, X, Z], mask: [Y, X, Z] bool
    """
    out = np.empty_like(vols_4d, dtype=np.float32)
    for c in range(vols_4d.shape[0]):
        v = vols_4d[c]
        m = mask
        if m.any():
            vals = v[m]
            mu = vals.mean()
            std = vals.std() + NORM_EPS
            vv = (v - mu) / std
        else:
            mu = v.mean()
            std = v.std() + NORM_EPS
            vv = (v - mu) / std
        out[c] = vv.astype(np.float32)
    return out


@torch.no_grad()
def _predict_logits(model, x, device):
    """
    Handles model outputs (tuple/list or tensor). Expects x: [B,C,D,H,W].
    """
    model.eval()
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _tta_logits(model, x, device):
    """
    Flip-based TTA. x: [B,C,D,H,W]. Returns averaged logits [B,4,D,H,W].
    Flips are in (D,H,W) index space -> dims (2,3,4) for torch.
    """
    if not USE_TTA:
        return _predict_logits(model, x, device)

    logits_list = []
    for dims in TTA_FLIPS:
        if dims is None:
            y = _predict_logits(model, x, device)
        else:
            xx = torch.flip(x, dims)
            y = _predict_logits(model, xx, device)
            y = torch.flip(y, dims)
        logits_list.append(y)
    return torch.stack(logits_list, 0).mean(0)


def _postprocess_wt(classmap, min_cc_vox=WT_MIN_CC_VOX, keep_topk=WT_KEEP_TOPK):
    """
    WT-focused cleanup:
      - Build WT mask (any tumor class > 0)
      - Remove connected components smaller than min_cc_vox
      - Optionally keep only top-K largest components
      - Apply cleaned WT as a mask on the class map (zero-out spurious islands)
    classmap: np.uint8 [Y,X,Z] with {0..3} labels (0=bg, 1=TC, 2=ED, 3=ET)
    """
    wt = (classmap > 0).astype(np.uint8)

    # Connected components on WT
    lab, num = ndi.label(wt)
    if num == 0:
        return classmap  # nothing to do

    # Component sizes
    sizes = ndi.sum(wt, lab, index=np.arange(1, num+1))
    sizes = np.asarray(sizes)

    # Filter by threshold
    keep_ids = [i+1 for i, s in enumerate(sizes) if s >= max(1, min_cc_vox)]

    # Optionally keep top-K by size
    if keep_topk > 0 and keep_ids:
        order = np.argsort(-sizes) + 1  # component labels sorted desc by size
        keep_ids = list(order[:keep_topk])

    cleaned_wt = np.isin(lab, keep_ids).astype(np.uint8)

    # Mask classmap by cleaned WT (remove small FPs while preserving class labels inside)
    out = classmap.copy()
    out[cleaned_wt == 0] = 0
    return out


def run_inference(model, input_root, output_root, device):
    """
    Runs inference over BraTS-style input tree:
      input_root/
        <case_id>/
          <case_id>-t1c.nii.gz
          <case_id>-t2w.nii.gz
          <case_id>-t2f.nii.gz
    Saves: output_root/<case_id>.nii.gz
    """
    torch.set_grad_enabled(False)
    model.eval()

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Determinism-ish for CUDA kernels
    torch.backends.cudnn.benchmark = False

    for case_dir in tqdm(sorted([p for p in input_root.iterdir() if p.is_dir()]), desc="Inference"):
        cid = case_dir.name

        try:
            # Load modalities with headers
            t1c_nii, t1c = _safe_load_mod(case_dir, MODS[0])
            t2w_nii, t2w = _safe_load_mod(case_dir, MODS[1])
            t2f_nii, t2f = _safe_load_mod(case_dir, MODS[2])
        except FileNotFoundError as e:
            print(f"⚠️ {e}. Skipping {cid}.")
            continue

        # use FLAIR header as reference (consistent with training)
        ref_nii = t2f_nii
        affine = ref_nii.affine
        header = ref_nii.header
        orig_shape = t1c.shape  # assume all same shape

        # Brain mask + margin bbox
        msk = brain_mask_from_nonzero(t1c, t2w, t2f)
        crop_slices = bbox_from_mask(
            msk, margin=BRAIN_MARGIN, shape=orig_shape)

        # If mask empty, emit zeros and move on
        if not msk.any():
            print(f"⚠️ Empty brain mask in {cid}, writing zeros.")
            pred_native = np.zeros(orig_shape, dtype=np.uint8)
        else:
            # Crop modalities to bbox
            cropped = np.stack([
                t1c[crop_slices], t2w[crop_slices], t2f[crop_slices]
            ], axis=0)  # [C,Y,X,Z]

            # Fit to target (center crop/pad)
            fitted_list, slicelist, padlist = [], None, None
            for c in range(cropped.shape[0]):
                arr_fit, sls, pds = fit_to_shape(cropped[c], TARGET_SHAPE)
                fitted_list.append(arr_fit)
                slicelist, padlist = sls, pds
            fitted = np.stack(fitted_list, 0)  # [C,Ty,Tx,Tz]

            # Z-score per modality using in-mask stats computed on cropped brain
            cropped_mask = msk[crop_slices]
            fitted_mask, _, _ = fit_to_shape(
                cropped_mask.astype(np.uint8), TARGET_SHAPE)
            fitted_mask = fitted_mask.astype(bool)
            fitted = zscore_per_mod_in_mask(fitted, fitted_mask)  # float32

            # To tensor (B,C,D,H,W) with dtype float32
            # NOTE: torch expects NCDHW -> here D=Y, H=X, W=Z? We'll stick to [B,C,Y,X,Z] consistently; model trained same way.
            tensor = torch.from_numpy(fitted)[None].to(device)

            # Sliding window + TTA
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = sliding_window_inference(
                    inputs=tensor,
                    roi_size=ROI_SIZE,
                    sw_batch_size=SW_BATCH,
                    predictor=lambda x: _tta_logits(model, x, device),
                    overlap=SW_OVERLAP,
                    mode="gaussian",
                    sw_device=device,
                    device="cpu" if device.type == "cuda" else device,
                )  # [1,4,Y,X,Z]

            # Argmax classmap
            pred_fit = logits.softmax(1).argmax(1).squeeze(
                0).cpu().numpy().astype(np.uint8)

            # WT-focused postproc
            pred_fit = _postprocess_wt(
                pred_fit, min_cc_vox=WT_MIN_CC_VOX, keep_topk=WT_KEEP_TOPK)

            # Map back to cropped space, then to native space
            pred_crop = invert_fit(
                pred_fit, cropped.shape[1:], slicelist, padlist)
            pred_native = np.zeros(orig_shape, dtype=np.uint8)
            pred_native[crop_slices] = pred_crop

            # Memory hygiene
            del tensor, logits, pred_fit
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Save
        out_img = nib.Nifti1Image(pred_native, affine, header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, output_root / f"{cid}.nii.gz")

    print("✅ Inference complete. Predictions saved to:", output_root)
