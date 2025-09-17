from monai.losses import DiceLoss, DiceFocalLoss, TverskyLoss

def getCriterion():
    return DiceLoss(softmax=True)
    # return DiceFocalLoss(softmax=True)
    # return TverskyLoss(softmax=True)
