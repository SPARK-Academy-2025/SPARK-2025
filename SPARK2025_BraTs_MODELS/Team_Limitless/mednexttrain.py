import subprocess
import os


class TrainingRunner:
    def __init__(self, task_id, nnnunet_raw, nnnunet_preprocessed, nnunet_results):
        self.task_id = task_id
        self.nnnunet_raw = nnnunet_raw
        self.nnnunet_preprocessed = nnnunet_preprocessed
        self.nnunet_results = nnunet_results

    def set_environment(self):
        print("Setting up environment variables for nnUNet...")
        os.environ['nnUNet_raw_data_base'] = self.nnnunet_raw
        os.environ['nnUNet_preprocessed'] = self.nnnunet_preprocessed
        os.environ['RESULTS_FOLDER'] = self.nnunet_results

    def preprocess(self, planner):
        print(f"Preprocessing data for task {self.task_id}...")
        # Simulate preprocessing step
        subprocess.run(
            f"mednextv1_plan_and_preprocess -t {self.task_id} -pl3d {planner}",
            shell=True,
            check=True
        )

    def train(self, trainer, plan, fold, weights=None):
        if weights:
            command = f"mednextv1_train 3d_fullres {trainer} {self.task_id} {fold} -p {plan} -pretrained_weights {weights}"
        else:
            command = f"mednextv1_train 3d_fullres {trainer} {self.task_id} {fold} -p {plan}"

        subprocess.run(
            command,
            shell=True,
            check=True
        )

    def run(self, planner, trainer, plan, fold=0, weights=None):
        try:
            self.preprocess(planner)
            self.train(trainer, plan, fold, weights)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the training: {e}")


def run_training(
        task_id,
        planner,
        trainer,
        plan,
        nnunet_raw,
        nnunet_preprocessed,
        nnunet_results,
        fold=0,
        weights=None,
        runner_class=TrainingRunner
        ):
    runner = runner_class(task_id, nnunet_raw, nnunet_preprocessed, nnunet_results)
    runner.set_environment()
    runner.run(planner, trainer, plan, fold, weights)


if __name__ == "__main__":
    # Example usage
    task_id = 444
    planner = "ExperimentPlanner3D_v21_customTargetSpacing_1x1x1"
    trainer = "nnUNetTrainerV2_MedNeXt_S_kernel5"
    plan = "nnUNetPlansv2.1_trgSp_1x1x1"
    nnunet_raw = "/home/spark17/TeamLimitless/experiments/mednext/raw"
    nnunet_preprocessed = "/home/spark17/TeamLimitless/experiments/mednext/preprocessed"
    nnunet_results = "/home/spark17/TeamLimitless/experiments/mednext/results"
    pretrained_weights = "/home/spark17/TeamLimitless/experiments/mednext/results/nnUNet/3d_fullres/Task444_BraTS-SSA-2025/nnUNetTrainerV2_MedNeXt_S_kernel5__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_best.model"
    run_training(task_id, planner, trainer, plan, nnunet_raw, nnunet_preprocessed, nnunet_results, fold=0, weights=pretrained_weights)