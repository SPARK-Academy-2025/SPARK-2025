import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.focal_loss import FocalLoss
from nnunetv2.training.loss.hausdorff import SoftHausdorffLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        
        # Adding individual losses for logging
        all_losses_dict = {
            "description": "deepsupervision not applied for individial losses",
            "all_losses": [
                                {
                                    "loss_name": "dice_loss",
                                    "loss": dc_loss,
                                    "loss_weight": self.weight_dice
                                },
                                {
                                    "loss_name": "ce_loss",
                                    "loss": ce_loss,
                                    "loss_weight": self.weight_ce
                                }
                            ]
                        }

        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        
        # Adding individual losses for logging
        all_losses_dict = {
            "description": "deepsupervision not applied for individial losses",
            "all_losses": [
                                {
                                    "loss_name": "dice_loss",
                                    "loss": dc_loss,
                                    "loss_weight": self.weight_dice
                                },
                                {
                                    "loss_name": "ce_loss",
                                    "loss": ce_loss,
                                    "loss_weight": self.weight_ce
                                }
                            ]
                        }

        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
       
        # Adding individual losses for logging
        all_losses_dict = {
            "description": "deepsupervision not applied for individial losses",
            "all_losses": [
                                {
                                    "loss_name": "dice_loss",
                                    "loss": dc_loss,
                                    "loss_weight": self.weight_dice
                                },
                                {
                                    "loss_name": "ce_loss",
                                    "loss": ce_loss,
                                    "loss_weight": self.weight_ce
                                }
                            ]
                        }

        return result

class DC_Focal_Hausdorff_Loss(nn.Module):
    """
    Combined loss: Soft Dice + Focal + Soft Hausdorff.
    Weights for each component need not sum to one.

    Args:
        dice_kwargs (dict): kwargs for SoftDiceLoss.
        focal_kwargs (dict): kwargs for FocalLoss.
        hausdorff_kwargs (dict): kwargs for SoftHausdorffLoss.
        weight_dice (float): multiplier for dice loss.
        weight_focal (float): multiplier for focal loss.
        weight_hausdorff (float): multiplier for hausdorff loss.
        ignore_label (int, optional): label value to ignore in loss.
        dice_class: SoftDiceLoss class or variant.
        focal_class: FocalLoss class or variant.
        hausdorff_class: SoftHausdorffLoss class or variant.
    """
    def __init__(self,
                 dice_kwargs: dict,
                 focal_kwargs: dict,
                 hausdorff_kwargs: dict,
                 weight_dice: float = 1.0,
                 weight_focal: float = 1.0,
                 weight_hausdorff: float = 1.0,
                 ignore_label: int = None,
                 dice_class=SoftDiceLoss,
                 focal_class=FocalLoss,
                 hausdorff_class=SoftHausdorffLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_hausdorff = weight_hausdorff
        self.ignore_label = ignore_label

        # instantiate components
        # dice and hausdorff expect softmaxed probabilities
        dice_kwargs = dice_kwargs.copy()
        hausdorff_kwargs = hausdorff_kwargs.copy()
        if ignore_label is not None:
            # ensure ignore_index passed to focal if needed
            focal_kwargs = focal_kwargs.copy()
            focal_kwargs['ignore_index'] = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **dice_kwargs)
        self.focal = focal_class(**focal_kwargs)
        self.hd = hausdorff_class(apply_nonlin=softmax_helper_dim1, **hausdorff_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        net_output: logits of shape (b, c, ...)
        target: label map of shape (b, ...) or one-hot (b, c, ...)
        """
        # Prepare mask for ignore_label
        mask = None
        if self.ignore_label is not None:
            mask = (target != self.ignore_label)
            target = torch.where(mask, target, torch.tensor(0, device=target.device))

        # Dice loss (requires one-hot target)
        dice_loss = self.dc(net_output, target.unsqueeze(1) if target.ndim+1==net_output.ndim else target, loss_mask=mask)

        # Focal loss (takes logits and class indices)
        focal_targets = target if target.ndim+1==net_output.ndim else target
        focal_loss = self.focal(net_output, focal_targets)

        # Hausdorff loss
        hausdorff_loss = self.hd(net_output, target)

        # Weighted sum
        result = (self.weight_dice * dice_loss
                 + self.weight_focal * focal_loss
                 + self.weight_hausdorff * hausdorff_loss)

        # logging individual losses
        self.last_losses = {
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'hausdorff_loss': hausdorff_loss
        }

        return result
