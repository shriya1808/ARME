#!/bin/bash -l
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --mem=64G

#SBATCH --mail-user=shriya.mittapalli@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV

WORKDIR="$WORK/ARME1"
cd "$WORKDIR" || exit 1
echo "Running in: $(pwd)"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

if [ ! -f ~/.bash_profile ]; then
  echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" > ~/.bash_profile
fi

module add python
conda config --add pkgs_dirs $WORK/software/private/conda/pkgs
conda config --add envs_dirs $WORK/software/private/conda/envs



conda config --set channel_priority strict

ENV_NAME="mimic-pipeline"
ENV_DIR="$WORK/software/private/conda/envs/$ENV_NAME"

# Create environment only if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Conda env not found. Creating $ENV_NAME ..."
    conda env create -f environment.yml -n $ENV_NAME
    conda activate $ENV_NAME
    conda update -n base -c conda-forge conda -y
    conda install -c conda-forge tqdm scikit-learn imbalanced-learn nltk -y
    conda install -c conda-forge ipywidgets pytorch -y
    pip install import-ipynb captum PyRuSH medspacy
    conda install -c conda-forge "jinja2=2.11.2" "markupsafe=2.0.1" -y
    conda install ipykernel -y
else
    echo "Conda env $ENV_NAME already exists."
    conda activate $ENV_NAME
fi

python3 readmission_main.py --workdir "$WORK/ARME1/" --datasetName "patient_features_CAD_train"

