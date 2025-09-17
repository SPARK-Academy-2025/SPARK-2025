import json
import shutil
import os
import SimpleITK as sitk
import numpy as np



def copy_BraTS_segmentation_and_convert_labels_to_nnUNet_SSA_2023(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    img_npy = img_npy.astype(int)

    uniques = np.unique(img_npy)
    for u in uniques:
        #print(f"value of u = {u}")
        if u not in [0, 1, 2, 3]:  #changed here since SSA has no 4 as label
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy) 
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 3] = 3 # incase it is already in 1, 2, 3 format
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    seg_new[img_npy < 0] = 0
    seg_new[img_npy > 4] = 0
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


# https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/dataset_conversion/utils.py

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)

def convert_to_nnUNet_MedNeXt_SSA_2023(source_file, destination_folder):
    os.makedirs(os.path.join(destination_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "labelsTr"), exist_ok=True)
    for folder in os.listdir(source_file):
        t1c_name = f"{folder}-t1c.nii.gz"
        t1c_path = os.path.join(source_file, folder, t1c_name)
        new_t1c_name = f"{folder}_0000.nii.gz"
        destination_t1c = os.path.join(destination_folder, "imagesTr", new_t1c_name)
        shutil.copy(t1c_path, destination_t1c)
        
        t1n_name = f"{folder}-t1n.nii.gz"
        t1n_path = os.path.join(source_file, folder, t1n_name)
        new_t1n_name = f"{folder}_0001.nii.gz"
        destination_t1n = os.path.join(destination_folder, "imagesTr", new_t1n_name)
        shutil.copy(t1n_path, destination_t1n)
        
        t2f_name = f"{folder}-t2f.nii.gz"
        t2f_path = os.path.join(source_file, folder, t2f_name)
        new_t2f_name = f"{folder}_0002.nii.gz"
        destination_t2f = os.path.join(destination_folder, "imagesTr", new_t2f_name)
        shutil.copy(t2f_path, destination_t2f)
        
        t2w_name = f"{folder}-t2w.nii.gz"
        t2w_path = os.path.join(source_file, folder, t2w_name)
        new_t2w_name = f"{folder}_0003.nii.gz"
        destination_t2w = os.path.join(destination_folder, "imagesTr", new_t2w_name)
        shutil.copy(t2w_path, destination_t2w)
        

        seg_name = f"{folder}-seg.nii.gz"
        seg_path = os.path.join(source_file, folder, seg_name)
        new_seg_name = f"{folder}.nii.gz"
        destination_seg = os.path.join(destination_folder, "labelsTr", new_seg_name)
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet_SSA_2023(in_file=seg_path, out_file=destination_seg)

# for raw dataset transfer for mednext
RAW_DATA_BASE = os.environ["nnUNet_raw_data_base"]

source_file = os.environ["ORIGINAL_TRAIN_DATA"] # Path to the source file
destination_folder =os.path.join(RAW_DATA_BASE,"nnUNet_raw_data", "Task1137_BraTS2023_SSA") # Destination folder path  
convert_to_nnUNet_MedNeXt_SSA_2023(source_file, destination_folder)


# for dataset.json
# For MedNeXt
channel_names= {
        "0": "t1c",
        "1": "t1",
        "2": "t2f",
        "3": "t2"
    }
labels = {
        0 : "Background",
        1 : "SNFH",
        2 : "NETC",
        3 : "ET ",
        4 : "RC"
    }

modalities = ('T1C', 'T1', 'T2', 'FLAIR')

generate_dataset_json(output_file = os.path.join(RAW_DATA_BASE, "nnUNet_raw_data/Task1137_BraTS2023_SSA/dataset.json"), 
                    imagesTr_dir = os.path.join(RAW_DATA_BASE, "nnUNet_raw_data/Task1137_BraTS2023_SSA/imagesTr"),
                    imagesTs_dir = None,
                    modalities = modalities,
                    labels = labels, 
                    dataset_name = "Task1137_BraTS2023_SSA",
                    dataset_description = "Task1137_BraTS2023_SSA dataset")
