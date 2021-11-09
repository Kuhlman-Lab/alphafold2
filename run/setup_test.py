import os
import numpy as np
import unittest

import setup

class TestQueryParsing(unittest.TestCase):

    def setUp(self) -> None:
        fasta_path = os.path.join('testdata', 'two_seq.fasta')
        a3m_path = os.path.join('testdata', 'two_seq.a3m')
        csv_path = os.path.join('testdata', 'four_seq.csv')
        seq1 = 'A'*30
        seq2 = 'C'*30
        seq3 = 'D'*30
        oligo1 = '2:1'
        oligo2 = '2'

        self.input_dir = 'testdata'
        self.files = {'fasta': [fasta_path], 'a3m': [a3m_path],
                      'csv': [csv_path]}
        self.monomer_queries = [(a3m_path, seq1), (fasta_path, seq1),
                                (fasta_path, seq2)]
        self.multimer_queries = [(csv_path, oligo1, [seq1, seq2]),
                                 (csv_path, oligo2, [seq3])]
        
    def test_QueryManager(self) -> None:
        qm = setup.QueryManager(input_dir=self.input_dir)

        self.assertEqual(qm.files, self.files)
        self.assertEqual(qm.others, [])

        qm.parse_files()

        self.assertEqual(qm.monomer_queries, self.monomer_queries)
        self.assertEqual(qm.multimer_queries, self.multimer_queries)
        
        
if __name__ == '__main__':
    unittest.main()
