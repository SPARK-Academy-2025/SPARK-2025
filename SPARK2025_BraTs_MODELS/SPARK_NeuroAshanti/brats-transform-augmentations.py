import os
import torch
import numpy as np
import torchio as tio
from pathlib import Path
from scipy.ndimage import binary_dilation

class LabelMaskedElasticDeformation(tio.Transform):
    def __init__(
        self,
        label_key: str = 'seg',
        dilation_radius: int = 16,
        num_control_points: int = 16,
        max_displacement: float = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label_key = label_key
        self.dilation_radius = dilation_radius
        self.elastic_transform = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement=max_displacement,
        )

    def apply_transform(self, subject):
        label_data = subject[self.label_key].data.squeeze().numpy()
        label_mask = label_data > 0

        if self.dilation_radius > 0:
            structure = np.ones((3, 3, 3))
            for _ in range(self.dilation_radius):
                label_mask = binary_dilation(label_mask, structure=structure)

        deformed_subject = self.elastic_transform(subject)

        for key, image in subject.items():
            if isinstance(image, (tio.ScalarImage, tio.LabelMap)):
                original_data = image.data
                deformed_data = deformed_subject[key].data
                mask_tensor = torch.from_numpy(label_mask).bool()
                mixed_data = torch.where(mask_tensor, deformed_data, original_data)
                deformed_subject[key].set_data(mixed_data)

        return deformed_subject

def augment_brats_data(input_dir, num_new_samples):
    
    input_dir = Path(input_dir)
    modalities = ['t1c', 't1n', 't2f', 't2w']

    
    subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS-SSA-')])

    subjects = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        seg_path = subject_dir / f'{subject_id}-seg.nii.gz'

        if not seg_path.exists():
            print(f"Skipping {subject_id}: Missing segmentation file.")
            continue

        subject_dict = {'seg': tio.LabelMap(seg_path)}
        valid_subject = True
        for modality in modalities:
            modality_path = subject_dir / f'{subject_id}-{modality}.nii.gz'
            if modality_path.exists():
                subject_dict[modality] = tio.ScalarImage(modality_path)
            else:
                print(f"Skipping {subject_id}: Missing {modality} file.")
                valid_subject = False
                break

        if valid_subject:
            subject = tio.Subject(**subject_dict)
            subjects.append((subject_id, subject))
        else:
            print(f"Skipping {subject_id}: Incomplete modalities.")
   

    
    transforms = tio.Compose([
        tio.RandomFlip(axes=('LR',), p=0.5),
        tio.RandomAffine(scales=(0.8, 1.2), degrees=10, translation=15, p=0.3),
        tio.RandomBiasField(coefficients=0.5, p=0.5),
        tio.RandomElasticDeformation(max_displacement=5, p=0.5),
        LabelMaskedElasticDeformation(
            label_key='seg',
            dilation_radius=16,
            max_displacement=12,
            p=1.0
        )
    ])

    existing_ids = [
        int(subject_id.split('-')[-2]) 
        for subject_id, _ in subjects 
        if subject_id.split('-')[-2].isdigit()
    ]
    last_id_num = max(existing_ids) if existing_ids else 0

    output_dir = Path("ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2-transform-augment")
    output_dir.mkdir(exist_ok=True)

    counter = 1
    for subject_id, subject in subjects:
        print(f"Processing: {subject_id}")
        for i in range(num_new_samples):
            new_id_num = last_id_num + counter
            new_subject_id = f'BraTS-SSA-{new_id_num:05d}-000'
            counter += 1

            transformed_subject = transforms(subject)

            subject_output_dir = output_dir / new_subject_id
            subject_output_dir.mkdir(exist_ok=True)

            transformed_subject['seg'].save(subject_output_dir / f'{new_subject_id}-seg.nii.gz')
            for modality in modalities:
                transformed_subject[modality].save(subject_output_dir / f'{new_subject_id}-{modality}.nii.gz')

            print(f"Generated and saved: {new_subject_id}")
        

if __name__ == "__main__":
    input_directory = "../ssa/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    number_of_samples = 5 
    augment_brats_data(input_directory, number_of_samples)