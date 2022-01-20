from distributor import Distributor
from run_af2 import af2_init, af2

import numpy as np
import random
from typing import Sequence

from utils.query_utils import generate_random_sequences

# List of 20 canonical amino acids.
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def naive_fitness(result):
    return np.mean(result['plddt'])


def test_distributor(
        n_workers: int,
        mode: str = 'monomer',
        arg_file: str = 'flags.txt'):

    n_workers = n_workers
    init_len = 25

    if mode == 'monomer':
        lengths = [init_len]
    else:
        lengths = [[init_len, init_len]]

    dist = Distributor(n_workers, af2_init, arg_file, lengths, naive_fitness)

    all_work = []
    all_results = []
    for _ in range(2):
        work_list = generate_random_sequences(lengths, 5)
        results = dist.churn(work_list)
        
        for seq in work_list:
            all_work.append(seq)
        for score in results:
            all_results.append(score)

    dist.spin_down()

    for i, j in zip(all_work, all_results):
        print('result:', i[0], j[0])


if __name__ == '__main__':
    
    arg_file = './testdata/flags_longleaf.txt'
    #arg_file = './testdata/flags.txt'

    print('Monomer Test:')
    test_distributor(2, 'monomer', arg_file)

    print('Multimer Test:')
    test_distributor(2, 'multimer', arg_file)

    #print('Multimer Gap Test:')
    #test_distributor('monomer', arg_file)
