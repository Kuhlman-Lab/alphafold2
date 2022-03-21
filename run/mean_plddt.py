from utils.utils import decompress_pickle
import numpy as np

if __name__ == '__main__':
    results_file = '/home/nzrandolph/git/alphafold/examples/monomer/outputs_final/sequence_0_model_1_ptm_0_results.pbz2'

    results = decompress_pickle(results_file)
    
    print(f'Mean pLDDT: {np.mean(results["plddt"])}')
    
