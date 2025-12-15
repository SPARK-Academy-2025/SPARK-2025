


import os

data_dir = "/home/spark17/TeamLimitless/experiments/segmamba/data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
val_data_dir = "/home/spark17/TeamLimitless/experiments/segmamba/data/raw_data/BraTS2023/BraTS2024-SSA-Challenge-ValidationData"

def rename_mri_data(data_dir):
    all_cases = os.listdir(data_dir)

    for case_name in all_cases:
        case_dir = os.path.join(data_dir, case_name)

        for data_name in os.listdir(case_dir):

            if "-" not in data_name:
                continue
            new_name = data_name.split("-")[-1]

            new_path = os.path.join(case_dir, new_name)

            old_path = os.path.join(case_dir, data_name)

            os.rename(old_path, new_path)

            print(f"{new_path} Naming success")

if __name__ == "__main__":
    if os.path.exists(data_dir):
        rename_mri_data(data_dir)
    if os.path.exists(val_data_dir):
        rename_mri_data(val_data_dir)


