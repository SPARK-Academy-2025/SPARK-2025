"""Enhanced postprocessing module with adaptive thresholds and morphology-aware processing"""

import numpy as np
import cc3d
from skimage.morphology import dilation, ball

def adaptive_rm_dust(pred_mat):
    """Enhanced dust removal with adaptive thresholds and morphology preservation"""
    # Initial conservative pass to preserve small structures
    pred_mat_clean = cc3d.dust(pred_mat, threshold=70, connectivity=26)
    
    # Secondary pass with label-specific processing
    for label in [1, 2, 3]:  # Process each label separately
        if label in pred_mat_clean:
            label_mask = (pred_mat_clean == label)
            
            # Label-specific thresholds
            thresholds = {1: 35, 2: 60, 3: 30}  # TC, WT, ET respectively
            structure = ball(1) if label == 3 else ball(2)  # Smaller for ET
            
            # Morphological closing to preserve structure
            label_mask = dilation(label_mask, structure)
            cleaned = cc3d.dust(label_mask, threshold=thresholds[label], connectivity=26)
            pred_mat_clean[np.logical_and(pred_mat_clean == label, ~cleaned)] = 0
    
    return pred_mat_clean

def get_tissue_wise_seg(pred_mat, tissue_type, dilation_size=0):
    """Enhanced with optional morphological dilation to bridge small gaps"""
    mask = np.zeros_like(pred_mat)
    
    if tissue_type == 'WT':
        mask = pred_mat > 0
    elif tissue_type == 'TC':
        mask = np.logical_or(pred_mat == 1, pred_mat == 3)
    elif tissue_type == 'ET':
        mask = pred_mat == 3
    
    if dilation_size > 0:
        mask = dilation(mask, ball(dilation_size))
    
    return mask.astype(np.uint16)

def enhanced_rm_tt_dust(pred_mat, tt):
    """Enhanced dust removal with adaptive morphology"""
    # Strategy parameters
    strategies = {
        'ET': {
            'threshold': 10,  # Very sensitive for ET
            'connectivity': 6,
            'dilation': 1,   # Small dilation to bridge gaps
            'min_volume': 10 # Minimum volume to preserve
        },
        'TC': {
            'threshold': 40,
            'connectivity': 18,
            'dilation': 0,
            'min_volume': 20
        },
        'WT': {
            'threshold': 60,
            'connectivity': 26,
            'dilation': 0,
            'min_volume': 30
        }
    }
    
    params = strategies[tt]
    pred_mat_tt = get_tissue_wise_seg(pred_mat, tt, params['dilation'])
    
    # Two-stage dust removal
    temp_clean = cc3d.dust(
        pred_mat_tt,
        threshold=params['min_volume'],
        connectivity=params['connectivity']
    )
    final_clean = cc3d.dust(
        temp_clean,
        threshold=params['threshold'],
        connectivity=params['connectivity']
    )
    
    rm_dust_mask = np.logical_and(pred_mat_tt==1, final_clean==0)
    pred_mat[rm_dust_mask] = 0
    return rm_dust_mask

def enhanced_fill_holes(pred_mat, tt, label, rm_dust_mask):
    """Enhanced hole filling with structure preservation"""
    hole_params = {
        'ET': {'threshold': 10, 'connectivity': 6, 'max_hole_size': 15},
        'TC': {'threshold': 20, 'connectivity': 18, 'max_hole_size': 30},
        'WT': {'threshold': 40, 'connectivity': 26, 'max_hole_size': 50}
    }
    
    params = hole_params[tt]
    pred_mat_tt = get_tissue_wise_seg(pred_mat, tt)
    
    # Detect holes more precisely
    tt_holes = 1 - pred_mat_tt
    tt_holes_rm = cc3d.dust(
        tt_holes,
        threshold=params['threshold'],
        connectivity=params['connectivity']
    )
    
    # Remove overly large holes
    large_holes = cc3d.dust(
        tt_holes_rm,
        threshold=params['max_hole_size'],
        connectivity=params['connectivity']
    )
    tt_holes_rm[large_holes > 0] = 0
    
    tt_filled = 1 - tt_holes_rm
    holes_mask = np.logical_and.reduce((
        tt_filled == 1,
        pred_mat == 0,
        rm_dust_mask,
        cc3d.dust(tt_filled, threshold=5, connectivity=6) > 0  # Ensure connected
    ))
    pred_mat[holes_mask] = label

def rm_dust_fh(pred_mat):
    """Optimized processing pipeline with ET-first strategy"""
    # Initial cleanup
    pred_mat = adaptive_rm_dust(pred_mat)
    
    # ET-specific processing
    rm_et_mask = enhanced_rm_tt_dust(pred_mat, 'ET')
    enhanced_fill_holes(pred_mat, 'TC', 1, rm_et_mask)
    
    # TC processing
    rm_tc_mask = enhanced_rm_tt_dust(pred_mat, 'TC')
    enhanced_fill_holes(pred_mat, 'WT', 2, rm_tc_mask)
    
    # Final WT processing
    _ = enhanced_rm_tt_dust(pred_mat, 'WT')
    
    # Final morphological smoothing
    for label in [1, 2, 3]:
        if label in pred_mat:
            label_mask = (pred_mat == label)
            pred_mat[label_mask & ~cc3d.dust(label_mask, threshold=10, connectivity=6)] = 0
    
    return pred_mat