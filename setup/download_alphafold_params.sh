#!/bin/bash
# Modified from https://github.com/deepmind/alphafold/blob/main/scripts/download_alphafold_params.sh

set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/params"
SOURCE_URL1="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
SOURCE_URL2="https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar"
SOURCE_URL3="https://storage.googleapis.com/alphafold/alphafold_params_2022-01-19.tar"

BASENAME=$(basename "${SOURCE_URL3}")

mkdir --parents "${ROOT_DIR}"
wget "${SOURCE_URL3}" --directory-prefix="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
  --directory="${ROOT_DIR}" --preserve-permissions
rm "${ROOT_DIR}/${BASENAME}"
mv "${ROOT_DIR}/params_model_1_multimer.npz" "${ROOT_DIR}/params_model_1_multimer_v1.npz"
mv "${ROOT_DIR}/params_model_2_multimer.npz" "${ROOT_DIR}/params_model_2_multimer_v1.npz"
mv "${ROOT_DIR}/params_model_3_multimer.npz" "${ROOT_DIR}/params_model_3_multimer_v1.npz"
mv "${ROOT_DIR}/params_model_4_multimer.npz" "${ROOT_DIR}/params_model_4_multimer_v1.npz"
mv "${ROOT_DIR}/params_model_5_multimer.npz" "${ROOT_DIR}/params_model_5_multimer_v1.npz"

BASENAME=$(basename "${SOURCE_URL2}")

mkdir --parents "${ROOT_DIR}"
wget "${SOURCE_URL2}" --directory-prefix="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
  --directory="${ROOT_DIR}" --preserve-permissions
rm "${ROOT_DIR}/${BASENAME}"

BASENAME=$(basename "${SOURCE_URL1}")

mkdir --parents "${ROOT_DIR}"
wget "${SOURCE_URL1}" --directory-prefix="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
  --directory="${ROOT_DIR}" --preserve-permissions
rm "${ROOT_DIR}/${BASENAME}"
