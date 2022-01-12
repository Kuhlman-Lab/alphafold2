from distributor import Distributor
from run_af2 import af2_init, af2

import numpy as np
import random
from typing import Sequence

AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def naive_fitness(result):
    return np.mean(result['plddt'])


def generate_random_monomers(length: int, num_seq: int, aalist=None) -> Sequence[Sequence[str]]:
    if aalist == None:
        aalist = AA_LIST

    return [[''.join(random.choices(aalist, k=length))] for _ in range(num_seq)]


def generate_random_multimers(lengths: Sequence[int], num_seq: int, aalist=None) -> Sequence[Sequence[Sequence[str]]]:
    if aalist == None:
        aalist = AA_LIST

    seqs_list = []
    for _ in range(num_seq):
        seqs = []
        for length in lengths:
            seq = ''.join(random.choices(aalist, k=length))
            seqs.append(seq)

        seqs_list.append([seqs])

    return seqs_list

def test_distributor(mode: str = 'monomer'):

    n_workers = 2
    init_len = 25

    if mode == 'monomer':
        lengths = [init_len]
    else:
        lengths = [[init_len, init_len]]

    dist = Distributor(n_workers, af2_init, 'flags.txt', lengths, naive_fitness)

    all_work = []
    all_results = []
    for _ in range(5):
        if mode == 'monomer':
            work_list = generate_random_monomers(lengths[0], 4)
        else:
            work_list = generate_random_multimers(lengths[0], 4)

        results = dist.churn(work_list)
        for seq in work_list:
            all_work.append(seq)
        for score in results:
            all_results.append(score)

    dist.spin_down()

    for i, j in zip(all_work, all_results):
        print('result:', i[0], j[0])


if __name__ == '__main__':

    print('Monomer Test:')
    test_distributor(mode='monomer')

    print('Multimer Test:')
    test_distributor(mode='multimer')
