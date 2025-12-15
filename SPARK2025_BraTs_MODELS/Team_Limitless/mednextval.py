import os
import subprocess
from mednexttrain import TrainingRunner


class ValidationRunner(TrainingRunner):
    
    def validate(self, checkpoint, dataset_path, output_path, trainer, plan):
        """
        Validate a model using the specified dataset and trainer.
        :param checkpoint: Name of the checkpoint model to use, defaults to model_final_checkpoint.
        :param dataset_path: Path to the dataset for validation.
        :param output_path: Path to save the validation results.
        :param trainer: The trainer to use for validation.
        :param plan: The training plan to use.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        postprocess_command = f'mednextv1_determine_postprocessing -m 3d_fullres -t {self.task_id} -tr {trainer} -p {plan}'
        command = f"mednextv1_predict -i {dataset_path} -o {output_path} -t {self.task_id} -f 'all' -tr {trainer} -p {plan} -chk {checkpoint}"
        print(f"Running validation with command: {command}")
        try:
            subprocess.run(f'{postprocess_command};{command}', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Validation failed with error: {e}")
        else:
            print(f"Validation completed successfully. Results saved to {output_path}")

    def run_validation(
            self,
            checkpoint="model_final_checkpoint",
            dataset_path=None,
            output_path=None,
            trainer="nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1",
            plan="nnUNetPlansv2.1_trgSp_1x1x1"
    ):
        """
        Run the validation process.
        :param checkpoint: Name of the checkpoint model to use, defaults to model_final_checkpoint.
        :param dataset_path: Path to the dataset for validation.
        :param output_path: Path to save the validation results.
        :param trainer: The trainer to use for validation.
        :param plan: The training plan to use.
        """
        self.set_environment()
        self.validate(
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            output_path=output_path,
            trainer=trainer,
            plan=plan
        )


if __name__ == "__main__":
    
    task_id = 444
    trainer = "nnUNetTrainerV2_MedNeXt_S_kernel5"
    plan = "nnUNetPlansv2.1_trgSp_1x1x1"
    nnunet_raw = "/home/spark17/TeamLimitless/experiments/mednext/raw"
    nnunet_preprocessed = "/home/spark17/TeamLimitless/experiments/mednext/preprocessed"
    nnunet_results = "/home/spark17/TeamLimitless/experiments/mednext/results"
    output_path = "/home/spark17/TeamLimitless/experiments/mednext/results/predictions/k5/fold-all"
    dataset_path = "/home/spark17/TeamLimitless/experiments/mednext/raw/nnUNet_raw_data/Task444_BraTS-SSA-2025/imagesVal"

    runner = ValidationRunner(task_id, nnunet_raw, nnunet_preprocessed, nnunet_results)

    runner.run_validation(
        checkpoint="model_best",
        dataset_path=dataset_path,
        output_path=output_path,
        trainer=trainer,
        plan=plan
    )

