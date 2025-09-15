#Sparkans Docker File

FROM --platform=linux/amd64 pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
LABEL authors="SPARK2025-TEAM-SPARKANS"

RUN apt-get update -y

COPY requirements.txt .
COPY networks networks/

RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt 

WORKDIR /opt/conda/bin/
RUN pip3 install -e /workspace/networks/medNeXt/
RUN pip3 install -e /workspace/networks/nnUNet/

# We may want to change these files or folders frequently, keep them here to avoid 
# downloading dependencies from scratch!
WORKDIR /workspace/

COPY models models/
COPY sparkans.sh prepare_nnunet.py prepare_mednext.py convert_mednext_npz.py cleanup.py ./

# Global environment variables
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"

# nnUNetv2 environment variables
ENV nnUNet_raw="/workspace/"
ENV nnUNet_preprocessed="/workspace/"
ENV nnUNet_results="/workspace/models"

# medNeXt environment variables
ENV nnUNet_raw_data_base="/workspace/"
ENV nnUNet_preprocessed="/workspace/"
ENV RESULTS_FOLDER="/workspace/models"

# Environment variables for going live (DO NOT MODIFY)
ENV INPUT_PATH="/input/"
ENV OUTPUT_PATH="/output/"
ENV TEMP_PATH="/tmp/"

# Other necessary instructions 
CMD ./sparkans.sh
