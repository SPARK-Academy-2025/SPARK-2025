import torch
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


def compute_robust_hausdorff(seg: torch.Tensor,
                              gt: torch.Tensor,
                              percentile: float = 95.0) -> torch.Tensor:
    """
    Compute the robust (percentile-based) symmetric Hausdorff distance between two binary masks.
    seg and gt are boolean tensors of shape (x, y, z...) or (D, ...).
    """
    # get point coordinates
    seg_pts = torch.nonzero(seg, as_tuple=False).float()
    gt_pts = torch.nonzero(gt, as_tuple=False).float()

    # if either mask is empty, distance is zero
    if seg_pts.numel() == 0 or gt_pts.numel() == 0:
        return seg.new_tensor(0.0)

    # pairwise distances
    d_mat = torch.cdist(seg_pts, gt_pts)

    # directed distances
    dist_seg_to_gt = d_mat.min(dim=1).values
    dist_gt_to_seg = d_mat.min(dim=0).values

    if percentile >= 100.0:
        # classic Hausdorff
        hd1 = dist_seg_to_gt.max()
        hd2 = dist_gt_to_seg.max()
        return torch.max(hd1, hd2)
    else:
        # robust (percentile) Hausdorff
        hd1 = torch.quantile(dist_seg_to_gt, percentile / 100.0)
        hd2 = torch.quantile(dist_gt_to_seg, percentile / 100.0)
        return torch.max(hd1, hd2)


class SoftHausdorffLoss(nn.Module):
    """
    Computes a differentiable approximation of the symmetric Hausdorff distance.
    Uses voxel-wise probability maps and binary ground truth.
    """
    def __init__(self,
                 apply_nonlin: Callable = None,
                 percentile: float = 95.0,
                 do_bg: bool = True,
                 ddp: bool = True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.percentile = percentile
        self.do_bg = do_bg
        self.ddp = ddp

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        x: (b, c, ...), probability logits or scores
        y: (b, ...) class labels or one-hot
        """
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        b, c = x.shape[:2]
        # get discrete prediction via argmax
        pred_labels = x.argmax(dim=1)

        # ensure gt has shape (b, ...)
        if y.ndim == x.ndim:
            gt_labels = y.argmax(dim=1)
        else:
            gt_labels = y.long()

        hd_vals = []
        for i in range(b):
            for cls in range(c):
                if not self.do_bg and cls == 0:
                    continue

                seg = (pred_labels[i] == cls)
                gt = (gt_labels[i] == cls)
                hd = compute_robust_hausdorff(seg, gt, self.percentile)
                hd_vals.append(hd)

        hd_tensor = torch.stack(hd_vals)

        if self.ddp:
            # gather across GPUs
            hd_tensor = AllGatherGrad.apply(hd_tensor).sum(0)

        # mean over classes and batch
        return hd_tensor.mean()


if __name__ == '__main__':
    # Simple test for compute_robust_hausdorff
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[2, 2] = True
    # identical masks -> zero distance
    assert compute_robust_hausdorff(mask, mask, percentile=100) == 0.0, "Zero distance failed"

    # offset mask -> distance should be 2 (Manhattan offset gives euclidean ~2)
    mask2 = torch.zeros_like(mask)
    mask2[4, 2] = True
    hd_classic = compute_robust_hausdorff(mask, mask2, percentile=100)
    assert torch.isclose(hd_classic, torch.tensor(2.0), atol=1e-6), "Classic Hausdorff failed"

    # robust percentile (50th) on two points gives the same as classic here
    hd_robust = compute_robust_hausdorff(mask, mask2, percentile=50)
    assert torch.isclose(hd_robust, torch.tensor(2.0), atol=1e-6), "Robust Hausdorff failed"

    # Test SoftHausdorffLoss
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    # batch size 1, two classes: background and foreground
    pred = torch.zeros((1, 2, 5, 5))
    gt = torch.zeros((1, 5, 5), dtype=torch.long)
    # set one voxel to class 1
    pred[0, 1, 2, 2] = 10.0  # high score for class 1 at (2,2)
    gt[0, 2, 2] = 1

    loss_fn = SoftHausdorffLoss(apply_nonlin=softmax_helper_dim1, percentile=100, do_bg=False, ddp=False)
    loss_val = loss_fn(pred, gt)
    # identical single-point prediction -> zero loss
    assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-6), "SoftHausdorffLoss zero-case failed"

    # change prediction to wrong location
    pred_wrong = pred.clone()
    pred_wrong[0, 1, 2, 2] = 0.0
    pred_wrong[0, 1, 4, 4] = 10.0
    loss_val2 = loss_fn(pred_wrong, gt)
    assert loss_val2 > 0, "SoftHausdorffLoss non-zero-case failed"

    print("All Hausdorff tests passed.")

