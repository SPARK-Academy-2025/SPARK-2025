
# ----------------------- Imports ----------------------- #

import os
import nibabel as nib
import matplotlib.pyplot as plt
import shutil
import json
import cv2
import numpy as np
import plotly.graph_objects as go
import subprocess

# ----------------------- Utilities ----------------------- #

def overlay(img_nii, mask_nii,
            class_colours: dict = {0: (0, 0, 0), 1: (200, 0, 0),
                                   2: (0, 200, 0), 3: (0, 0, 200)},
            alpha = 1, beta = 0.7):

    img = img_nii.get_fdata()
    mask = mask_nii.get_fdata()
    mask = np.rint(mask).astype(np.uint8)

    norm_img = 255 * (img - np.min(img)) / (np.ptp(img) + 1e-8)
    norm_img_rgb = np.stack([norm_img] * 3, axis=-1)
    overlay = norm_img_rgb.copy()

    for label, colour in class_colours.items():
        mask_region = mask == label
        if np.any(mask_region):
            colour = np.array(colour, dtype=np.uint8)
            # this was really intresting but .clip ensures that no values that are calculated are going out of the ranges
            # this is important so that 367 doesn't become 100 which would give wrong colourings.
            overlay[mask_region] = (alpha * norm_img_rgb[mask_region] + beta * colour).clip(0, 255).astype(np.uint8)

    return nib.Nifti1Image(overlay, affine=img_nii.affine)

def localize(img_nii, mask_nii, colour = (0, 0, 255), thickness = 2):
  img = img_nii.get_fdata()
  mask = mask_nii.get_fdata()
  mask = np.rint(mask).astype(np.uint8)

  normalized_img = (255 * (img - np.min(img)) / (np.ptp(img) + 1e-8)).astype(np.uint8)

  rgb_img = np.stack([normalized_img] * 3, axis=-1)
  overlay = rgb_img.copy()

  x = []
  y = []
  w = []
  h = []

  for z in range(mask.shape[2]):
    mask_slice = mask[:, :, z]
    rgb_img_slice = overlay[:, :, z]
    rgb_img_slice = np.ascontiguousarray(rgb_img_slice) #apparently: OpenCV's C++ backend requires the image to be contiguous in memory so that it fits
    if not np.any(mask_slice):
      x.insert(z, 0)
      y.insert(z, 0)
      w.insert(z, 0)
      h.insert(z, 0)
    else:
      #hierarchy is for parent-child relationships of contours. might be useful later for specefic bounding boxes
      contours, hierarchy = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
          continue # there was a small dot that appeared that turned out to be a contour so this filters those out
        current_x, current_y, current_w, current_h = cv2.boundingRect(cnt)
        x.insert(z, current_x)
        y.insert(z, current_y)
        w.insert(z, current_w)
        h.insert(z, current_h)
        cv2.rectangle(rgb_img_slice, (current_x, current_y), (current_x + current_w, current_y + current_h), colour, thickness)
      overlay[:, :, z, :] = rgb_img_slice


  overlay_nii = nib.Nifti1Image(overlay, affine=img_nii.affine)
  return overlay_nii, tuple([x, y, w, h])

def find_black_masks(truth_mask_nii, pred_mask_nii):
  truth_mask = truth_mask_nii.get_fdata()
  pred_mask = pred_mask_nii.get_fdata()
  skips = []

  for i in range(truth_mask.shape[2]):
    truth_mask_slice = truth_mask[:, :, i]
    pred_mask_slice = pred_mask[:, :, i]
    if not np.any(truth_mask_slice) and not np.any(pred_mask_slice):
        skips.append(i)

  return skips

def localized_IOU(img_nii, truth_mask_nii, pred_mask_nii, black_masks: bool = False):
  _ , coordinates_truth = localize(img_nii, truth_mask_nii)
  _ , coordinates_pred = localize(img_nii, pred_mask_nii)

  ious_per_slice = []

  if black_masks:
    skips = find_black_masks(truth_mask_nii, pred_mask_nii)
  else:
    skips = []

  for i in range(0, 155, 1):
    if i in skips:
       continue
    #Getting Intersection top left (a) and bottom right (d)
    intersect_a_x = max(coordinates_truth[0][i], coordinates_pred[0][i])
    intersect_a_y = max(coordinates_truth[1][i], coordinates_pred[1][i])
    intersect_d_x = min((coordinates_truth[0][i] + coordinates_truth[2][i]), (coordinates_pred[0][i] + coordinates_pred[2][i]))
    intersect_d_y = min((coordinates_truth[1][i] + coordinates_truth[3][i]), (coordinates_pred[1][i] + coordinates_pred[3][i]))

    #Areas Calculation
    intersect_area = abs(intersect_d_y - intersect_a_y) * abs(intersect_d_x - intersect_a_x)
    truth_area = abs(coordinates_truth[2][i] * coordinates_truth[3][i])
    pred_area = abs(coordinates_pred[2][i] * coordinates_pred[3][i])
    union_area = truth_area + pred_area - intersect_area + 1e-8

    ious_per_slice.append(intersect_area/union_area)

  return sum(ious_per_slice)/len(ious_per_slice)

def volume_plot(img_nii, title=None):
    img = img_nii.get_fdata().astype(np.uint8)
    fig = go.Figure(data=go.Volume(
            #this section is wildly confusing but here is what happens:
            ##creates a single x point for 255 which is the dimension length then repeats it to become 1D datapoints to 3D datapoints based on y and x
            x=np.arange(img.shape[0]).repeat(img.shape[1] * img.shape[2]),
            #creates 1D datapoints of y then repeats over z making a 2D plane, then redoing the whole thing per slice of x
            y=np.tile(np.arange(img.shape[1]).repeat(img.shape[2]), img.shape[0]),
            #creates 1D datapoints of z, then repeats that for both x and y
            z=np.tile(np.arange(img.shape[2]), img.shape[0] * img.shape[1]),
            value=img.flatten(),
            isomin=0.5,
            isomax=3,
            opacity=1,
            surface_count=1,
            colorscale='pinkyl'
        ))

    fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            )
        )

    fig.show()

# ----------------------- Big-Boy Function ----------------------- #

def nnunet_infer(dataset_path: str,
               output_path: str,
               nnunet_path: str,
               checkpoint_dir: str,
               dataset_name: str = "Dataset001_Inference",
               dataset_id: int = 1,
               dataset_description: str = "Nothing to say here"
               ):
    # -------------- Dataset Preparation -------------- #
    # mappings between nnUNet file names and case names
    mapping = {}
    mapping_count = 0

    for folder in sorted(os.listdir(dataset_path)):
        added_mapping = False
        if "DS_Store" in folder: continue
        for scanfile in os.listdir(f"{dataset_path}/{folder}"):
            if "t1n" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "t1c" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "t2w" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "t2f" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
        if added_mapping:
            mapping_count += 1

    print(f"Length of Dataset is:  {mapping_count}")

    # -------------- nnUNet setup -------------- #
    ### nnUNet_raw:
    base_raw = f"{nnunet_path}/nnUNet_raw/{dataset_name}"
    base_imagesTs = f"{base_raw}/imagesTs"
    # base_imagesTr = f"{base_raw}/imagesTr"
    # base_labelsTr = f"{base_raw}/labelsTr"
    os.makedirs(base_raw, exist_ok=True)
    os.makedirs(base_imagesTs, exist_ok=True)
    # os.makedirs(base_imagesTr, exist_ok=True) # just to avoid assertion errors
    # os.makedirs(base_labelsTr, exist_ok=True) # just to avoid assertion errors

    """
    0000 = t1n
    0001 = t1c
    0002 = t2w
    0003 = t2f
    """

    for folder in sorted(os.listdir(dataset_path)):
        if "DS_Store" in folder: continue
        for scanfile in os.listdir(f"{dataset_path}/{folder}"):
            if "t1n" in scanfile:
                file_name = f"{mapping[folder]}_0000.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTs, file_name))
            if "t1c" in scanfile:
                file_name = f"{mapping[folder]}_0001.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTs, file_name))
            if "t2w" in scanfile:
                file_name = f"{mapping[folder]}_0002.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTs, file_name))
            if "t2f" in scanfile:
                file_name = f"{mapping[folder]}_0003.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTs, file_name))

    print(f"nnUNet_raw/{dataset_name}/imagesTs is set up ")

    ## dataset.json:
    dataset_json = {
    "name": f"{dataset_name}",
    "description": f"{dataset_description}",
    "channel_names": {
        "0": "t1n",
        "1": "t1c",
        "2": "t2w",
        "3": "t2f"
    },
    "labels": {
        "background" : 0,
        "whole" : 1,
        "enhancing tumor" : 2,
        "core" : 3
    },
    "file_ending": ".nii.gz",
    "numTraining": 0,
    "numTest": mapping_count
    }

    with open(os.path.join(base_raw, "dataset.json"), 'w') as json_filepath:
       json.dump(dataset_json, json_filepath)

    print("dataset.json is set up")
    print(f"nnUNet_raw/{dataset_name} is fully set up")

    # -------------- preparation of model -------------- #

    ## Setting up required enviroment variables
    os.environ["nnUNet_raw"] = f"{nnunet_path}/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = f"{nnunet_path}/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = f"{nnunet_path}/nnUNet_results"

    ## Creating directories
    os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_results"], exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print("Environment variables set.")

    # -------------- MyQuickTrainer.py -------------- #
    #MyQuickTrainer.py Set Up
    """
    Unfortunatley since i trained this 1000 epoch model using MyQuickTrainer the checkpoints is dependent on it
    Therfore the following section of creating MyQuickTrainer.py is needed which is quite unneccesary but here we are
    """

    template = """

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import numpy as np

class MyQuickTrainer(nnUNetTrainer):
    def __init__(   self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                    device: torch.device = torch.device('cuda')
                ):

        super().__init__(plans, configuration, fold, dataset_json, device)

    """
    trainer_dir = f"{nnunet_path}/nnunetv2/training/nnUNetTrainer"

    try:
        os.remove(f"{trainer_dir}/MyQuickTrainer.py")
    except FileNotFoundError:
       pass

    with open(f"{trainer_dir}/MyQuickTrainer.py", 'w') as f:
        f.write(f"{template}")

    # Checkpoint setup:
    checkpoint_to_dir = f"{nnunet_path}/nnUNet_results/{dataset_name}/MyQuickTrainer__nnUNetPlans__3d_fullres"
    json_from_path = f"{base_raw}/dataset.json"
    json_to_path = f"{checkpoint_dir}/dataset.json"

    shutil.copytree(checkpoint_dir, checkpoint_to_dir, dirs_exist_ok=True)
    os.remove(json_to_path)
    shutil.copy(json_from_path, json_to_path)
    print("Checkpoint and dataset.json Uploaded to nnUNet_results Successfully")
    print("Model Infering, Please wait....")

    # -------------- Inference -------------- #

    try:
        os.makedirs(f"{output_path}/non_mapped", exist_ok=True)
        inference_return_code = subprocess.call(["nnUNetv2_predict", "-i", f"{base_imagesTs}", "-o", f"{output_path}/non_mapped", "-d", f"{dataset_name}", "-c", "3d_fullres", "-f", "0", "-tr", "MyQuickTrainer"])
        if inference_return_code == 0:
          print("Inference Done Sucessfully and Non-mapped masks outputted")
          print("Will Remap filenames to case names now...")
        else:
           print("Inference didn't work properly")
    except Exception as err:
       print(f"\n{err}\n")

    # -------------- Remapping filename to Cases -------------- #

    for file in sorted(os.listdir(f"{output_path}/non_mapped")):
       case_name = f"{mapping.keys(file[:-7])}-seg.nii.gz"
       shutil.copyfile(f"{output_path}/non_mapped/{file}", f"{output_path}/{case_name}")

    reverse_mapping = {v: k for k, v in mapping.items()}

    for file in sorted(os.listdir(f"{output_path}/non_mapped")):
      for img_name, original_name in reverse_mapping.items():
            if file.startswith(img_name):
                src = os.path.join(f"{output_path}/non_mapped", file)
                new_filename = file.replace(f"{img_name}", original_name)
                dst = os.path.join(f"{output_path}", new_filename)
                shutil.copy(src, dst)


    shutil.rmtree(f"{output_path}/non_mapped")

    with open(f"{output_path}/mappings.json", 'w') as f:
      json.dump(mapping, f)

    print("Predictions Successfully Mapped and non_mapped is deleted. Mappings have been saved as well")
    print("---END---")

def nnunet_train(dataset_path: str, 
               output_path: str, 
               nnunet_path: str,
               checkpoint_dir: str = None,
               trainer_template: str = None,
               dataset_name: str = "Dataset001_Inference",
               dataset_id: int = 1,
               dataset_description: str = "Nothing to say here"
               ):
   
    # -------------- Dataset Preparation -------------- #
    # mappings between nnUNet file names and case names
    mapping = {}
    mapping_count = 0

    for folder in sorted(os.listdir(dataset_path)):
        added_mapping = False
        if "DS_Store" in folder: continue
        for scanfile in os.listdir(f"{dataset_path}/{folder}"):
            if "t1n" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "t1c" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "t2w" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "t2f" in scanfile:
                if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
            if "seg" in scanfile: 
               if not added_mapping:
                    mapping[folder] = f"img_{mapping_count:04d}"
                    added_mapping = True
        if added_mapping:
            mapping_count += 1

    print(f"Length of Dataset is:  {mapping_count}\n")

    # -------------- nnUNet setup -------------- # 
    ### nnUNet_raw: 
    base_raw = f"{nnunet_path}/nnUNet_raw/{dataset_name}"
    base_imagesTr = f"{base_raw}/imagesTr"
    base_labelsTr = f"{base_raw}/labelsTr"
    os.makedirs(base_raw, exist_ok=True)
    os.makedirs(base_imagesTr, exist_ok=True) 
    os.makedirs(base_labelsTr, exist_ok=True) 

    """ 
    0000 = t1n
    0001 = t1c
    0002 = t2w
    0003 = t2f
    """

    for folder in sorted(os.listdir(dataset_path)):
        if "DS_Store" in folder: continue
        for scanfile in os.listdir(f"{dataset_path}/{folder}"):
            if "t1n" in scanfile:
                file_name = f"{mapping[folder]}_0000.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTr, file_name))
            if "t1c" in scanfile:
                file_name = f"{mapping[folder]}_0001.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTr, file_name))
            if "t2w" in scanfile:
                file_name = f"{mapping[folder]}_0002.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTr, file_name))
            if "t2f" in scanfile:
                file_name = f"{mapping[folder]}_0003.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_imagesTr, file_name))
            if "seg" in scanfile:
                file_name = f"{mapping[folder]}.nii.gz"
                shutil.copy(os.path.join(f"{dataset_path}/{folder}", scanfile), os.path.join(base_labelsTr, file_name))

    
    print(f"nnUNet_raw/{dataset_name}/imagesTr is set up \n")
    print(f"nnUNet_raw/{dataset_name}/labelsTr is set up \n")

    ## dataset.json: 
    dataset_json = {
    "name": f"{dataset_name}",
    "description": f"{dataset_description}",
    "channel_names": {
        "0": "t1n",
        "1": "t1c",
        "2": "t2w",
        "3": "t2f" 
    },
    "labels": {
        "background" : 0,
        "whole" : 1,
        "enhancing tumor" : 2,
        "core" : 3
    },
    "file_ending": ".nii.gz",
    "numTraining": mapping_count,
    }

    with open(os.path.join(base_raw, "dataset.json"), 'w') as json_filepath: 
       json.dump(dataset_json, json_filepath)

    print("dataset.json is set up\n")
    print(f"nnUNet_raw/{dataset_name} is fully set up\n")

    # -------------- preparation of model -------------- #

    ## Setting up required enviroment variables
    os.environ["nnUNet_raw"] = f"{nnunet_path}/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = f"{nnunet_path}/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = f"{nnunet_path}/nnUNet_results"

    ## Creating directories
    os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

    print("Environment variables set.")

    # -------------- MyQuickTrainer.py -------------- #

    if trainer_template == None:
      template = """
    
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import numpy as np

class MyQuickTrainer(nnUNetTrainer):
    def __init__(   self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                    device: torch.device = torch.device('cuda')
                ):

        super().__init__(plans, configuration, fold, dataset_json, device)
    
    """
      trainer_dir = "nnUNet/nnunetv2/training/nnUNetTrainer"

      try: 
          os.remove(f"{trainer_dir}/MyQuickTrainer.py")
      except FileNotFoundError:
          pass

      with open(f"{trainer_dir}/MyQuickTrainer.py", 'w') as f:
          f.write(f"{template}")
    else: 
        trainer_dir = "nnUNet/nnunetv2/training/nnUNetTrainer"

        try: 
            os.remove(f"{trainer_dir}/MyQuickTrainer.py")
        except FileNotFoundError:
          pass

        with open(f"{trainer_dir}/MyQuickTrainer.py", 'w') as f:
            f.write(f"{template}")
    
    # Checkpoint setup: 
    if checkpoint_dir is not None: 
      checkpoint_to_dir = f"{nnunet_path}/nnUNet_results/{dataset_name}/MyQuickTrainer__nnUNetPlans__3d_fullres"
      json_from_path = f"{base_raw}/dataset.json"
      json_to_path = f"{checkpoint_dir}/dataset.json"

      shutil.copytree(checkpoint_dir, checkpoint_to_dir, dirs_exist_ok=True)
      os.remove(json_to_path)
      shutil.copy(json_from_path, json_to_path)
      print("Checkpoint and dataset.json Uploaded to nnUNet_results Successfully\n")

    # -------------- Data Prepocessing -------------- #
    try: 
      preprocessing_return_code = subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", f"{dataset_id}", "--verify_dataset_integrity"])
      if preprocessing_return_code.returncode == 0:
        print("Preprocessing done Successfully")
        print("Will start Training Now")
      else: 
        print("Preprocessing didn't work properly")
    except Exception as err: 
       print(f"{err}")

    # -------------- Training -------------- #
    os.environ["RESULTS_FOLDER"] = output_path
    # !nnUNetv2_train 5 3d_fullres 0 -tr MyQuickTrainer -device cuda # --c
    try:
        if checkpoint_dir is None: 
          training_return_code = subprocess.run(["nnUNetv2_train", f"{dataset_id} ", "3d_fullres", "0", "-tr", "MyQuickTrainer", "-device", "cuda"])
        else: 
          training_return_code = subprocess.run(["nnUNetv2_train", f"{dataset_id} ", "3d_fullres", "0", "-tr", "MyQuickTrainer", "-device", "cuda", "--c"])
        if training_return_code.returncode == 0:
          print("Training Done Sucessfully and Non-mapped masks outputted\n")
          print("Will Remap filenames to case names now...\n")
        else: 
           print("Training didn't work properly\n")
    except Exception as err: 
       print(f"\n{err}\n")

    return mapping
        
   



    


