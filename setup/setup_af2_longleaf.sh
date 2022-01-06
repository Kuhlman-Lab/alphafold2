#!/bin/bash

has_conda=$(grep /nas/longleaf/apps/anaconda/2020.07 ~/.bashrc)
if [ ! "$has_conda" ]; then
    echo "# >>> conda initialize >>>" >> ~/.bashrc
    echo "# !! Contents within this block are managed by 'conda init' !!" >> ~/.bashrc
    echo "__conda_setup='\$('/nas/longleaf/apps/anaconda/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)'" >> ~/.bashrc
    echo "if [ \$? -eq 0 ]; then" >> ~/.bashrc
    echo "    eval '\$__conda_setup'" >> ~/.bashrc
    echo "else" >> ~/.bashrc
    echo "    if [ -f '/nas/longleaf/apps/anaconda/2020.07/etc/profile.d/conda.sh' ]; then" >> ~/.bashrc
    echo "        . '/nas/longleaf/apps/anaconda/2020.07/etc/profile.d/conda.sh'" >> ~/.bashrc
    echo "    else" >> ~/.bashrc
    echo "        export PATH='/nas/longleaf/apps/anaconda/2020.07/bin:\$PATH'" >> ~/.bashrc
    echo "    fi" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
    echo "unset __conda_setup" >> ~/.bashrc
    echo "# <<< conda initialize <<<" >> ~/.bashrc

    echo "Anaconda has been set up."
else
    echo "Anaconda has already been initialized for you."
fi

has_cuda_dir=$(grep CUDA_DIR=/nas/longleaf/apps/cuda/11.2 ~/.bashrc)
if [ ! "$has_cuda_dir" ]; then
    echo "export CUDA_DIR=/nas/longleaf/apps/cuda/11.2" >> ~/.bashrc
    echo "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/nas/longleaf/apps/cuda/11.2" >> ~/.bashrc
    echo "CUDA_DIR environmental variable set."
else
    echo "CUDA_DIR variable is already set for you."
fi

has_xla_flags=$(grep XLA_FLAGS=--xla_gpu_cuda_data_dir=/nas/longleaf/apps/cuda/11.2 ~/.bashrc)
if [ ! "$has_xla_flags" ]; then
    echo "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/nas/longleaf/apps/cuda/11.2" >> ~/.bashrc
    echo "XLA_FLAGS environmental variable set."
else
    echo "XLA_FLAGS variable is already set for you."
fi

source ~/.bashrc

conda config --add envs_dirs /nas/longleaf/home/nzrandol/.conda/envs

echo "Set up script has completed. Make sure to 'source ~/.bashrc' before attempting to run AF2."
