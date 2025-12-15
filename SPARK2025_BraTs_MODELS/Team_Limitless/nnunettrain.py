import subprocess
import os
from glob import glob
from zipfile import ZipFile
import shutil

"""
Plan:
- Set the nnunet environment variables for training
- run the command to do preprocessing
- run the command to do training
"""
class BaselineTrainer:
    def __init__(
        self,
        dataset_path,
        nnunet_raw_dir,
        nnunet_preprocessed_dir,
        nnunet_results_dir
        ):
        """
        Initialize the BaselineTrainer with dataset and nnUNet directories.
        The dataset_path should contain the training and validation datasets in zipped files.
        """

        assert os.path.exists(dataset_path), f"Dataset Path {dataset_path} does not exist."
        assert os.path.isdir(nnunet_raw_dir), f"nnUNet raw directory {nnunet_raw_dir} does not exist."
        assert os.path.isdir(nnunet_preprocessed_dir), f"nnUNet preprocessed directory {nnunet_preprocessed_dir} does not exist."
        assert os.path.isdir(nnunet_results_dir), f"nnUNet results directory {nnunet_results_dir} does not exist."

        self.dataset_path = dataset_path
        self.nnunet_raw_dir = nnunet_raw_dir
        self.nnunet_preprocessed_dir = nnunet_preprocessed_dir
        self.nnunet_results_dir = nnunet_results_dir

    def set_environment_variables(self):
        """
        Set the nnUNet environment variables for training.
        This method should be implemented by child classes to set preprocessing and results directories.
        """
        os.environ['nnUNet_raw'] = self.nnunet_raw_dir
        os.environ['nnUNet_preprocessed'] = self.nnunet_preprocessed_dir
        os.environ['nnUNet_results'] = self.nnunet_results_dir

        print("Environment variables set for nnUNet.")
        print(f"nnUNet raw data base: {self.nnunet_raw_dir}")
        print(f"nnUNet preprocessed data: {self.nnunet_preprocessed_dir}")
        print(f"nnUNet results folder: {self.nnunet_results_dir}")

    
    def unzip_or_move_files(self):
        # Check if path is a zip file
        if self.dataset_path == self.nnunet_raw_dir:
            print("Dataset path matches nnunet raw directory")
            return
        elif self.dataset_path.split('/')[-1] in os.listdir(self.nnunet_raw_dir):
            print(f"{self.nnunet_raw_dir} already contains dataset named {self.dataset_path.split('/')[-1]}")
            return
        elif self.dataset_path.endswith('.zip'):
            print(f"Dataset path is a zip file, proceeding to extract to {self.nnunet_raw_dir}")
            ZipFile(self.dataset_path).extractall(self.nnunet_raw_dir)
        elif os.path.isdir(self.dataset_path):
            print(f"Dataset path is not a zip file, copying to {self.nnunet_raw_dir}")
            shutil.copytree(
                self.dataset_path,
                os.path.join(self.nnunet_raw_dir, self.dataset_path.split('/')[-1])
            )
        else:
            raise Exception(f"{self.dataset_path} is neither a directory nor a zip file")

    def prepare_dataset_converter(
        self,
        path,
        dataset_name='BRATS-SSA-2025',
        dataset_id=444
        ):
        """
        parameters:
            - path: str, the path where the nnUNet cloned repository
            - dataset_name: str, the name of the dataset to be used in nnUNet
            - dataset_id: int, the id of the dataset to be used in nnUNet
        """
        assert os.path.isdir(path), "Path must be a directory"
        filepath = f'{path}/nnunetv2/dataset_conversion/Dataset226_BraTS2024-BraTS-GLI.py'
        assert os.path.isfile(filepath), "Config file to edit not found"
        
        database_raw_name = self.dataset_path.split('/')[-1]
        if database_raw_name.endswith('.zip'):
            database_raw_name = database_raw_name.split('.')
        dataset_final_path = os.path.join(self.nnunet_raw_dir, database_raw_name)  
        with open(filepath, 'r') as f:
            lines = f.readlines()
            print(len(lines))
        with open(filepath, 'w') as f:
            for line in lines:
                if line.strip().startswith("extracted_BraTS2024_GLI_dir ="):
                    f.write(f"    extracted_BraTS2024_GLI_dir = '{dataset_final_path}'\n")
                elif line.strip().startswith("nnunet_dataset_name ="):
                    f.write(f"    nnunet_dataset_name = '{dataset_name}'\n")
                elif line.strip().startswith("nnunet_dataset_id ="):
                    f.write(f"    nnunet_dataset_id = {dataset_id}\n")
                else:
                    f.write(line)
        edited_lines = []
        edited_lines.append(subprocess.run(f'grep -h extracted_BraTS2024_GLI_dir\ = {filepath}', shell=True, capture_output=True))
        edited_lines.append(subprocess.run(f'grep -h nnunet_dataset_name\ = {filepath}', shell=True, capture_output=True))
        edited_lines.append(subprocess.run(f'grep -h nnunet_dataset_id\ = {filepath}', shell=True, capture_output=True))
        for line in edited_lines:
            print(line.stdout.decode().strip())
        return filepath

    def generate_dataset_config(self, path):
        assert os.path.isfile(path), 'Dataset converter file must be a file'
        subprocess.run(f'python3 {path}', shell=True, capture_output=True)

    def run_preprocessing(
        self,
        dataset_id=444,
        planner_name='nnUNetPlannerResEncM'
        ):
        """
        Run the nnUNet preprocessing command.
        This method should be implemented by child classes to run the preprocessing command.
        """
        command = f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity -pl {planner_name}"
        print(f"Running preprocessing command: {command}")
        result = subprocess.run(command, shell=True, capture_output=True)
        print(result.stdout.decode().strip())

    def run_training(
        self,
        dataset_id=444,
        trainer_name='nnUNetTrainerV2',
        pretrained_path=None,
        folds='all',
        device='cuda',
        plan='nnUNetResEncUNetMPlans'
        ):
        """
        Run the nnUNet training command.
        This method should be implemented by child classes to run the training command.
        """
        if pretrained_path:
            assert os.path.exists(pretrained_path), f"Pretrained path {pretrained_path} does not exist."
            command = f"nnUNetv2_train {dataset_id} 3d_fullres {folds} -device {device} -p {plan} -pretrained_weights {pretrained_path} --npz"
        else:
            command = f"nnUNetv2_train {dataset_id} 3d_fullres {folds} -device {device} -p {plan} --npz"
        print(f"Running training command: {command}")
        result = subprocess.run(command, shell=True, capture_output=True)
        if result.returncode != 0:
            print(f"Training command failed with error: {result.stderr.decode()}")
        else:
            print(f"Training command completed successfully: {result.stdout.decode()}")

    def run(
        self,
        path,
        dataset_name='BRATS-SSA-2025',
        dataset_id=444,
        planner_name='nnUNetPlannerResEncM',
        trainer_name='nnUNetTrainerV2',
        folds='all',
        device='cuda',
        plan='nnUNetResEncUNetMPlans'
        ):
        self.set_environment_variables()
        self.unzip_or_move_files()
        converter_path = self.prepare_dataset_converter(
            path=path,
            dataset_name=dataset_name,
            dataset_id=dataset_id
            )
        print(f"Converter path: {converter_path}")
        self.generate_dataset_config(converter_path)
        self.run_preprocessing(
            dataset_id=dataset_id,
            planner_name=planner_name
            )
        self.run_training(
            dataset_id=dataset_id,
            trainer_name=trainer_name,
            pretrained_path='/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_results/Dataset444_BRATS-SSA-2025/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_1/checkpoint_best.pth',
            folds=folds,
            device=device,
            plan=plan
            )




if __name__ == "__main__":
    path = '/home/spark17/TeamLimitless/experiments/nnunet/nnUNet'
    dataset_dir = '/home/spark17/scratch/spark17/data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'
    nnunet_raw_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_raw'
    nnunet_preprocessed_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_preprocessed'
    nnunet_results_dir = '/home/spark17/TeamLimitless/experiments/nnunet/baseline/nnUNet_results'

    trainer = BaselineTrainer(
        dataset_path=dataset_dir,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir
    )
    trainer.run(
        path=path,
        dataset_name='BRATS-SSA-2025',
        dataset_id=444,
        planner_name='nnUNetPlannerResEncM',
        trainer_name='nnUNetTrainerV2',
        folds=1,
        device='cuda',
        plan='nnUNetResEncUNetMPlans'
    )
    print("Training completed successfully.")

        



