#!/bin/bash

has_bashrc_updates=$(grep /home/nzrandol/anaconda3/etc/profile.d/conda.sh ~/.bashrc)
if [ ! "$has_bashrc_updates" ]; then
    echo "# >>> conda initialize >>>" >> ~/.bashrc
    echo "# !! Contents within this block are managed by 'conda init' !!" >> ~/.bashrc
    echo "__conda_setup='$('/home/nzrandol/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)'" >> ~/.bashrc
    echo "if [ $? -eq 0 ]; then" >> ~/.bashrc
    echo "    eval '$__conda_setup'" >> ~/.bashrc
    echo "else" >> ~/.bashrc
    echo "    if [ -f '/home/nzrandol/anaconda3/etc/profile.d/conda.sh' ]; then" >> ~/.bashrc
    echo "        . '/home/nzrandol/anaconda3/etc/profile.d/conda.sh'" >> ~/.bashrc
    echo "    else" >> ~/.bashrc
    echo "        export PATH='/home/nzrandol/anaconda3/bin:$PATH'" >> ~/.bashrc
    echo "    fi" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
    echo "unset __conda_setup" >> ~/.bashrc
    echo "# <<< conda initialize <<<" >> ~/.bashrc

    echo "AF2 has been set up. Make sure to 'source ~/.bashrc' before trying to activate the 'af2' conda env."
else
    echo "AF2 conda env has already been set up for you."
fi

has_cuda_dir=$(grep CUDA_DIR=/usr/local/cuda ~/.bashrc)
if [ ! "$has_cuda_dir" ]; then
    echo "export CUDA_DIR=/usr/local/cuda" >> ~/.bashrc

    echo "CUDA_DIR environmental variable set. Make sure to 'source ~/.bashrc' before trying to run AF2."
else
    echo "CUDA_DIR variable already set for you."
fi
