from nnunettrain import BaselineTrainer
import os
import shutil
from zipfile import ZipFile
import subprocess

class BaselineValidator(BaselineTrainer):

    def __init__(
        self,
        dataset_path,
        nnunet_raw_dir,
        nnunet_preprocessed_dir,
        nnunet_results_dir
        ):
        
        super().__init__(
            dataset_path=dataset_path,
            nnunet_raw_dir=nnunet_raw_dir,
            nnunet_preprocessed_dir=nnunet_preprocessed_dir,
            nnunet_results_dir=nnunet_results_dir
        )
    
    def unzip_or_move_files(self):
        # Check if path is a zip file
        path = os.path.join(self.nnunet_raw_dir, self.dataset_path.split('/')[-1].replace('.zip', ''))
        if self.dataset_path == self.nnunet_raw_dir:
            print("Dataset path matches nnunet raw directory")
            return
        elif os.path.isdir(path) and len(os.listdir(path)) > 0:
            print(f"{self.nnunet_raw_dir} already contains dataset named {self.dataset_path.split('/')[-1]}")
            return
        # elif self.dataset_path.endswith('.zip'):
        #     print(f"Dataset path is a zip file, proceeding to extract to {self.nnunet_raw_dir}")
        #     zip = ZipFile(self.dataset_path)
        #     extractables = filter(
        #         lambda x: x.endswith('.nii.gz'),
        #         zip.namelist()
        #     )
        #     os.makedirs(path, exist_ok=True)
        #     for file in extractables:
        #         zip.extract(os.path.basename(file), path)
        #     zip.close()
        #     print(f"Extracted files to {path}")
            
        elif os.path.isdir(self.dataset_path):
            print(f"Dataset path is not a zip file, copying to {self.nnunet_raw_dir}")
            os.makedirs(path, exist_ok=True)
            subdirs = os.listdir(self.dataset_path)
            for subdir in subdirs:
                subdir_path = os.path.join(self.dataset_path, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith('.nii.gz'):
                            shutil.copy(os.path.join(subdir_path, file), path)
            
        else:
            raise Exception(f"{self.dataset_path} is neither a directory nor a zip file")

    def replace_modality_names_by_ids(self, valpath, modality_dict):
        for file in os.listdir(valpath):
            if file.endswith('.nii.gz'):
                for modality, modality_id in modality_dict.items():
                    if modality in file.lower():
                        new_file_name = file.replace(f'-{modality}', f'_{modality_id}')
                        os.rename(
                            os.path.join(valpath, file),
                            os.path.join(valpath, new_file_name)
                        )

        


    def run_inference(
            self,
            dataset_path,
            output_dir,
            dataset_id,
            folds,
            device,
            plan,
            config_name,
            checkpoint
        ):
        command = f'nnUNetv2_predict -i {dataset_path} -o {output_dir} -p {plan} -d {dataset_id} -c {config_name} -device {device} -chk {checkpoint} --save_probabilities'
        print(f"Running inference with command: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True
        )

    def run(
        self,
        valpath,
        output_dir,
        dataset_id=444,
        folds='all',
        device='cuda',
        plan='nnUNetResEncUNetMPlans',
        config_name='3d_fullres',
        checkpoint='checkpoint_best.pth'
    ):
        self.set_environment_variables()
        self.unzip_or_move_files()
        self.replace_modality_names_by_ids(valpath, {
            't1n': '0000',
            "t1c": '0001',
            "t2w": '0002',
            "t2f": '0003'
        })
        self.run_inference(
            dataset_path=valpath,
            output_dir=output_dir,
            dataset_id=dataset_id,
            folds=folds,
            device=device,
            plan=plan,
            config_name=config_name,
            checkpoint=checkpoint
        )



if __name__ == "__main__":
    valpath = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_raw/BraTS2024-SSA-Challenge-ValidationData'
    output_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/predictions'
    dataset_dir = '/home/spark17/scratch/spark17/data/BraTS2024-SSA-Challenge-ValidationData'
    nnunet_raw_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_raw'

    nnunet_preprocessed_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_preprocessed'
    nnunet_results_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_results'

    validator = BaselineValidator(
        dataset_path=dataset_dir,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir
    )
    validator.run(
        valpath=valpath,
        output_dir=output_dir,
        dataset_id=444,
        folds='all',
        device='cuda',
        plan='nnUNetResEncUNetMPlans'
    )
    print("Training completed successfully.")

        



