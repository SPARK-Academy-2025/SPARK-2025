from mednextval import ValidationRunner
from nnunetval import BaselineValidator
from staple import StapleEnsambler
from segmambaval import SegMambaValRunner
import os

class EnsembleValidator:
    def __init__(self, task_id, input_dir):
        self.task_id = task_id
        self.dataset_name = os.path.basename(input_dir)
        # Make intermediate directories
        nnunetdir, mednextdir, segmambadir = self.setup_directories()
        self.copy_dataset(
            source=os.path.join(input_dir),
            destination=os.path.join(mednextdir, 'raw')
        )
        self.copy_dataset(
            source=os.path.join(input_dir),
            destination=os.path.join(segmamba_dir, 'raw')
        )
        # NNUnet already copies the dataset to the right place. No need to redo it.

        # Initialize validators
        self.nnunet_validator = BaselineValidator(
            dataset_path=input_dir,
            nnunet_raw_dir=os.path.join(nnunetdir, 'raw'),
            nnunet_preprocessed_dir=os.path.join(nnunetdir, 'preprocessed'),
            nnunet_results_dir=os.path.join(nnunetdir, 'results')
        )
        self.mednext_validator = ValidationRunner(
            task_id=task_id,
            nnnunet_raw=os.path.join(mednextdir, 'raw'),
            nnnunet_preprocessed=os.path.join(mednextdir, 'preprocessed'),
            nnunet_results=os.path.join(mednextdir, 'results')
        )
        self.segmamba_validator = SegMambaValRunner(
            base_dir=os.path.join(segmambadir, 'raw'),
            image_dir=self.dataset_name,
            preprocessed_dir=os.path.join(segmambadir, 'preprocessed'),
            output_dir=os.path.join(segmambadir, 'results'),
            model_path=os.path.join(segmambadir, 'model.pth')
        )
            image_dir=os.path.join(segmambadir, 'raw', self.dataset_name),

    def validate(self):
        # Run nnUNet validation
        self.nnunet_validator.run_validation(
            dataset_path=f'{nnunetdir}/raw/{self.dataset_name}',
            output_path='/tmp/nnunet/results/predictions',
            dataset_id=self.task_id,
        )
        # Run MedNeXt validation
        self.mednext_validator.run_validation(
            dataset_path=f'{mednextdir}/raw/{self.dataset_name}',
            output_path='/tmp/mednext/results/predictions',
            checkpoint='model_best',
        )
        self.segmamba_validator.run()

    def setup_directories(self):
        # create nnunet directories
        nnunetdir = '/tmp/nnunet'
        nnunet_raw = os.path.join(nnunetdir, 'raw')
        nnunet_preprocessed = os.path.join(nnunetdir, 'preprocessed')
        nnunet_results = os.path.join(nnunetdir, 'results')

        os.makedirs(nnunet_raw, exist_ok=True)
        os.makedirs(nnunet_preprocessed, exist_ok=True)
        os.makedirs(nnunet_results, exist_ok=True)

        # create mednext directories
        mednextdir = '/tmp/mednext'
        mednext_raw = os.path.join(mednextdir, 'raw')
        mednext_preprocessed = os.path.join(mednextdir, 'preprocessed')
        mednext_results = os.path.join(mednextdir, 'results')

        os.makedirs(mednext_raw, exist_ok=True)
        os.makedirs(mednext_preprocessed, exist_ok=True)
        os.makedirs(mednext_results, exist_ok=True)

        # Create segmamba directories
        segmambadir = '/tmp/segmamba'
        segmamba_raw = os.path.join(segmambadir, 'raw')
        segmamba_preprocessed = os.path.join(segmambadir, 'preprocessed')
        segmamba_results = os.path.join(segmambadir, 'results')

        os.makedirs(segmamba_raw, exist_ok=True)
        os.makedirs(segmamba_preprocessed, exist_ok=True)
        os.makedirs(segmamba_results, exist_ok=True)

        return [nnunetdir, mednextdir, segmambadir]

    def copy_dataset(self, source, destination):
        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)
        if os.path.isdir(source):
            for item in os.listdir(source):
                src_path = os.path.join(source, item)
                dst_path = os.path.join(destination, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
        else:
            raise ValueError(f"Source {source} is not a directory.")


def main(input_dir, output_dir):
    task_id = 444
    validator = EnsembleValidator(task_id, input_dir)
    validator.validate()
    # Run STAPLE ensambler
    staple_ensambler = StapleEnsambler(
        os.path.join('/tmp/nnunet', 'results', 'predictions'),
        os.path.join('/tmp/mednext', 'results', 'predictions'),
        os.path.join('/tmp/segmamba', 'results', 'predictions'),
    )
    staple_ensambler.run_and_save(output_dir)
    print(f"Validation completed for task {task_id}. Results saved to {output_dir}.")
    
