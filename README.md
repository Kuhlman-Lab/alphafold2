<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Steps to set up a local installation of our modified version of AlphaFold2.

### Prerequisites

Installation of Anaconda is required to load dependencies.
* Installing conda: [conda-install-link]

### Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/Kuhlman-Lab/alphafold.git
   ```
2. Load AlphaFold2 model weights from source using script: https://github.com/Kuhlman-Lab/alphafold/blob/main/setup/download_alphafold_params.sh 

3. Set up conda environment:
   ```sh
   conda env create -n alphafold -f setup/af2_env.yml
   pip3 install --upgrade jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   python3 -m pip install /path/to/alphafold/alphafold/
   ```

<!-- USAGE EXAMPLES -->
## Usage

The input file should be a CSV file. Each new line in the CSV will be identified as a separate query and will generate an individual prediction. Each line should contain a sequence starting with a comma, and each chain is separated by a comma. 

Find below the flags that can be provided to the model runner, grouped by functionality:

Input/Output
| Flag                  | Type   | Default | Description                                                                  |
| --------------------- | ------ | ------- | ---------------------------------------------------------------------------- |
| `--input_dir`         | `str`  | `''`    | Directory containing input files.                                            |
| `--output_dir`        | `str`  | `''`    | Directory to store prediction results. Default is `./prediction_{datetime}`. |
| `--params_dir`        | `str`  | `''`    | Path to parameter weights directory. Default is `../alphafold/data/`.        |
| `--compress_output`   | `flag` | `False` | Compress result dictionaries per query.                                      |
| `--dont_write_pdbs`   | `flag` | `False` | Skip writing PDB files.                                                      |
| `--save_timing`       | `flag` | `False` | Save runtime metrics as a `.pkl` file.                                       |
| `--batch_predictions` | `flag` | `False` | Enable batched predictions.                                                  |
| `--no_logging`        | `flag` | `False` | Disable logging output.                                                      |
| `--design_run`        | `flag` | `False` | Suppress most outputs unless overridden.                                     |
| `--initial_guess`     | `str`  | `None`  | Path to initial guess PDB file.                                              |

Sequence control
| Flag                    | Type  | Default | Description                                                                  |
| ----------------------- | ----- | ------- | ---------------------------------------------------------------------------- |
| `--min_length`          | `int` | `1`     | Minimum chain length for prediction.                                         |
| `--max_pad_size`        | `int` | `None`  | Pad short sequences to this length to avoid recompilation.                   |
| `--max_length`          | `int` | `2500`  | Maximum single-chain length.                                                 |
| `--max_multimer_length` | `int` | `2500`  | Maximum total multimer length. Note: validation limited above 1536 residues. |

MSA generation
| Flag                     | Type    | Default         | Description                                                                       |
| ------------------------ | ------- | --------------- | -----------------------------------------------------------------                 |
| `--msa_mode`             | `str`   | `'MMseqs2-U+E'` | MSA mode. Options: `MMseqs2-U+E`, `MMseqs2-U`, `single_sequence`.                 |
| `--custom_msa_path`      | `str`   | `None`          | Path to custom `.a3m` files. msa_mode should be single_sequence to use this flag. |
| `--insert_msa_gaps`      | `flag`  | `False`         | Add gaps to custom MSAs based on sequence differences.                            |
| `--update_msa_query_seq` | `float` | `1.0`           | Sequence identity threshold for updating custom MSAs.                             |

Template usage
| Flag                              | Type   | Default        | Description                                    |
| --------------------------------- | ------ | -------------- | ---------------------------------------------- |
| `--use_templates`                 | `flag` | `False`        | Use templates from MMseqs2.                    |
| `--max_template_date`             | `str`  | `'2100-01-01'` | Max release date for templates (`YYYY-MM-DD`). |
| `--custom_template_path`          | `str`  | `None`         | Path to custom template PDBs.                  |
| `--rm_template_seq`               | `flag` | `False`        | Remove sequences from templates.               |
| `--dont_mask_template_interchain` | `flag` | `False`        | Enable interchain template usage.              |
| `--permute_templates`             | `flag` | `False`        | Enable symmetric chain template permutation.   |

Model control
| Flag                   | Type    | Default | Description                                            |
| ---------------------- | ------- | ------- | ------------------------------------------------------ |
| `--use_ptm`            | `flag`  | `False` | Use pTM models for predicted aligned error (PAE).      |
| `--no_multimer_models` | `flag`  | `False` | Use monomer models even for multimers.                 |
| `--use_multimer_v1`    | `flag`  | `False` | Use original v1 multimer weights.                      |
| `--use_multimer_v2`    | `flag`  | `False` | Use v2 multimer weights (instead of v3).               |
| `--num_models`         | `int`   | `5`     | Number of models to run (1–5).                         |
| `--num_ensemble`       | `int`   | `1`     | Number of MSA ensemble runs per model (CASP14 used 8). |
| `--random_seed`        | `int`   | `None`  | Random seed for reproducibility.                       |
| `--num_seeds`          | `int`   | `1`     | Number of seeds per model/sequence.                    |
| `--is_training`        | `flag`  | `False` | Enable dropout to sample diverse structures.           |
| `--max_recycle`        | `int`   | `3`     | Max recycling steps for refinement.                    |
| `--recycle_tol`        | `float` | `0.0`   | Early stop recycling if Ca-RMS change < tol.           |

Other options
| Flag          | Type   | Default | Description                                                     |
| ------------- | ------ | ------- | --------------------------------------------------------------- |
| `--use_amber` | `flag` | `False` | Run Amber relaxation (improves geometry but increases runtime). |
| `--__override_disable` | `flag` | `False` | Internal override for debugging/disabling features.    |


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running and Analysis

To make a prediction, provide the sequences in the input CSV file. Set the options in the af2.flags file. Then, run the commands:
```sh
conda activate alphafold
python /path/to/alphafold/run/run_af2.py @af2.flags
   ```

Running the code will produce an outputs directory containing PDB files with predicted structures and .pbz2 files with compressed information about the predictions such as confidence metrics.
To generate pLDDT plots and PAE heatmaps, navigate to a directory containing .pbz2 files and run the command:
```sh
conda activate alphafold
python /path/to/alphafold/run/plots.py
   ```

## Specific examples

Specify your AF2 options in af2.flags.  

To turn off MSA generation:  
```sh
--msa_mode single_sequence  
   ```

To use precomputed MSA:  
```sh
--msa_mode single_sequence  
--custom_msa_path /path/to/directory/with/a3m/files/
   ```

To use custom templates:
```sh
--use_templates  
--custom_template_path templates
   ```

Since we have modified the original AF2 code to allow for custom template databases, you will have to make sure each .pdb file within the templates folder has a name consisting of 4 letters and numbers, essentially mimicking a file from the PDB database with a PDB code (although the file name does not actually have to be a real PDB code). Some examples could be “temp.pdb”, “1uzg.pdb”, “PPPP.pdb”, etc.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[conda-install-link]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html  
