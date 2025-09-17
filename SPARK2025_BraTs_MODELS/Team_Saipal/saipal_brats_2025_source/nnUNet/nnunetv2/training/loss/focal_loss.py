import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.
    Based on: https://arxiv.org/abs/1708.02002

    Args:
        gamma (float): focusing parameter gamma >= 0
        alpha (float or list[float], optional): balance factor. If a single float, applied to the positive class; "
                             "if list, must have length = number of classes.
        reduction (str): 'none' | 'mean' | 'sum'
        ignore_index (int, optional): specifies a target value that is ignored
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha=None,
                 reduction: str = 'mean',
                 ignore_index: int = None):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum'), "reduction must be 'none', 'mean', or 'sum'"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(self.alpha, (list, tuple)):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float)
        elif isinstance(self.alpha, float):
            # two-class case: alpha for class 1, (1 - alpha) for class 0
            self.alpha = torch.tensor([1 - self.alpha, self.alpha], dtype=torch.float)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: logits of shape (N, C, ...) or (N, C)
        targets: ground truth labels of shape (N, ...) or (N,)
        """
        # compute log probabilities
        log_prob = F.log_softmax(inputs, dim=1)
        prob = log_prob.exp()

        # gather log_prob and prob values at target labels
        targets = targets.long()
        if targets.dim() + 1 == inputs.dim():
            # add channel dim for gather
            targets = targets.unsqueeze(1)
        log_pt = torch.gather(log_prob, dim=1, index=targets).squeeze(1)
        pt = torch.gather(prob, dim=1, index=targets).squeeze(1)

        # apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # alpha for each sample
            if self.alpha.dim() > 1:
                # per-class alpha
                at = self.alpha.index_select(0, targets.squeeze(1))
            else:
                # binary alpha
                at = self.alpha[targets.squeeze(1)]
            log_pt = log_pt * at

        # focal factor
        loss = -((1 - pt) ** self.gamma) * log_pt

        # ignore index
        if self.ignore_index is not None:
            mask = targets.squeeze(1) != self.ignore_index
            loss = loss * mask

        # reduction
        if self.reduction == 'mean':
            # avoid dividing by zero when all masked
            if self.ignore_index is not None:
                denom = mask.sum().clamp_min(1)
                return loss.sum() / denom
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == '__main__':
    # Test 1: gamma=0 (should equal cross-entropy)
    fl1 = FocalLoss(gamma=0, alpha=None, reduction='mean')
    logits1 = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    labels1 = torch.tensor([0, 1])
    ce = F.cross_entropy(logits1, labels1)
    assert torch.allclose(fl1(logits1, labels1), ce, atol=1e-6), "FocalLoss vs CE mismatch"

    # Test 2: gamma>0 should reduce loss value relative to CE
    fl2 = FocalLoss(gamma=2.0, alpha=None, reduction='mean')
    val2 = fl2(logits1, labels1)
    assert val2 < ce, "FocalLoss not lower than CE for gamma>0"

    # Test 3: alpha balancing for binary classification
    fl3 = FocalLoss(gamma=0, alpha=0.25, reduction='none')
    loss3 = fl3(logits1, labels1)
    p0 = F.softmax(logits1, dim=1)[0, 0]
    manual0 = -torch.log(p0) * 0.75
    assert torch.allclose(loss3[0], manual0, atol=1e-6), "Alpha weighting mismatch"

    # Test 4: ignore_index
    fl4 = FocalLoss(gamma=0, alpha=None, reduction='sum', ignore_index=1)
    logits4 = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    labels4 = torch.tensor([1, 0])
    p1 = F.softmax(logits4, dim=1)[1, 0]
    expected4 = -torch.log(p1)
    assert torch.allclose(fl4(logits4, labels4), expected4, atol=1e-6), "Ignore index test failed"

    print("All FocalLoss tests passed.")

