#!/bin/bash
set -e

# Function to check if conda is installed
check_conda_installed() {
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
    else
        echo "Conda is not installed. Installing Miniconda..."
        install_miniconda
    fi
}


# Function to install Miniconda
install_miniconda() {
    echo "Downloading Miniconda installer..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest.sh
    echo "Running Miniconda installer..."
    bash Miniconda3-latest.sh -b -p $HOME/miniconda
    echo "Initializing Miniconda..."
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
}

# Function to check if conda environment exists
check_conda_env() {
    if conda env list | grep -q "dnlp_bart_generation"; then
        echo "Conda environment 'dnlp' already exists."
    else
        echo "Conda environment 'dnlp_bart_generation' does not exist. Creating environment..."
        conda create -n dnlp_bart_generation python=3.10 -y
    fi
}

# Main script execution
check_conda_installed
check_conda_env

# Initialize Conda for the current shell
eval "$(conda shell.bash hook)"

echo "Activating conda environment 'dnlp_bart_generation'..."
conda activate dnlp_bart_generation

echo $CONDA_DEFAULT_ENV

# Install packages
conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y tqdm==4.66.2 pandas==2.1.4 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 -c conda-forge -c huggingface
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0 nltk==3.9.1

# Download model on login-node

python - <<EOF
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
EOF

python - <<EOF
from transformers import AutoTokenizer, AutoModel, BartModel

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
model = BartModel.from_pretrained('facebook/bart-base')
EOF


python - <<EOF

import nltk
nltk.download('wordnet')

EOF
