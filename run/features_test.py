import os
import shutil
import unittest

import features

class TestRawInputs(unittest.TestCase):

    def setUp(self) -> None:
        fasta_path = os.path.join('testdata', 'two_seq.fasta')
        a3m_path = os.path.join('testdata', 'two_seq.a3m')
        csv_path = os.path.join('testdata', 'four_seq.csv')
        with open(a3m_path, 'r') as f:
            a3m_lines = f.read()
        self.true_a3m_a3m_lines = {a3m_path: a3m_lines}
        
        self.seq1 = 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHF'
        seq1_a3m_path = os.path.join('testdata', 'seq1_a3m.txt')
        seq1_templates_path = os.path.join('testdata', 'seq1_templates.txt')
        with open(seq1_a3m_path, 'r') as f:
            self.true_seq1_a3m_lines = f.read()
        with open(seq1_templates_path, 'r') as f:
            self.true_seq1_templates = f.read().strip()

        self.seq2 = 'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRV'
        seq2_a3m_path = os.path.join('testdata', 'seq2_a3m.txt')
        seq2_templates_path = os.path.join('testdata', 'seq2_templates.txt')
        with open(seq2_a3m_path, 'r') as f:
            self.true_seq2_a3m_lines = f.read()
        with open(seq2_templates_path, 'r') as f:
            self.true_seq2_templates = f.read().strip()

        oligo1 = '1:1'

        self.monomer_queries = [(fasta_path, self.seq1), (a3m_path, self.seq1)]
        self.multimer_queries = [(csv_path, oligo1, [self.seq1, self.seq2])]

    def test_raw_inputs(self) -> None:
        raw_inputs, a3m_lines = features.getMonomerRawInputs(
            monomer_queries=self.monomer_queries,
            use_env=True,
            use_templates=True)

        seq1_templates = ','.join(os.listdir(raw_inputs[self.seq1][1]))

        self.assertEqual(a3m_lines, self.true_a3m_a3m_lines)
        self.assertEqual(raw_inputs[self.seq1][0], self.true_seq1_a3m_lines)
        self.assertEqual(seq1_templates, self.true_seq1_templates)

        raw_inputs = features.getMultimerRawInputs(
            multimer_queries=self.multimer_queries,
            use_env=True,
            use_templates=True,
            raw_inputs=raw_inputs)

        seq2_templates = ','.join(os.listdir(raw_inputs[self.seq2][1]))

        self.assertEqual(raw_inputs[self.seq2][0], self.true_seq2_a3m_lines)
        self.assertEqual(seq2_templates, self.true_seq2_templates)

        self.assertEqual(len(raw_inputs.keys()), 2)

    def tearDown(self) -> None:
        shutil.rmtree('mmseqs2')
        
if __name__ == '__main__':
    unittest.main()
