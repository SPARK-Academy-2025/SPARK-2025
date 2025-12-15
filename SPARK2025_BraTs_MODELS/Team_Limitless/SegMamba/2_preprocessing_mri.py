
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = ["t2w.nii.gz",
                 "t2f.nii.gz",
                 "t1n.nii.gz",
                 "t1c.nii.gz"]
seg_filename = "seg.nii.gz"

base_dir = "/home/spark17/TeamLimitless/experiments/segmamba/data/raw_data/BraTS2023/"
image_dir = "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
val_image_dir = "BraTS2024-SSA-Challenge-ValidationData"

output_dir = "/home/spark17/TeamLimitless/experiments/segmamba/data/fullres/train/"

def process_train(
        output_dir=output_dir,
        image_dir=image_dir
        ):
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
        
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def plan(image_dir=image_dir):
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()


if __name__ == "__main__":

    import os

    if os.path.exists(os.path.join(base_dir, image_dir)):
        plan()
        process_train()