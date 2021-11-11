"""
Methods and classes for setting up AlphaFold. Includes argument parsing and 
input query parsing.
"""

import os
import argparse
from typing import Sequence
from utils import query_utils

class FileArgumentParser(argparse.ArgumentParser):
    """Overwrites default ArgumentParser to better handle flag files."""

    def convert_arg_line_to_args(self, arg_line: str) -> Sequence[str]:
        # Read from files where each line contains a flag and its value, e.g.
        # '--flag value'.
        split_line = arg_line.split(' ')

        if len(split_line) > 1:
            return [split_line[0], ' '.join(split_line[1:])]
        else:
            return split_line

        
def getAF2Parser() -> FileArgumentParser:
    """Gets an FileArgumentParser with necessary arguments to run AF2."""

    parser = FileArgumentParser(description='AF2 runner script that can take '
                                'command-line arguments or read from an '
                                'argument flag file.',
                                fromfile_prefix_chars='@')

    # I/O Arguments
    parser.add_argument('--input_dir',
                        default='.',
                        type=str,
                        help='Path to directory that contains input files. '
                        'Default is ./.')

    parser.add_argument('--output_dir',
                        default='',
                        type=str,
                        help='Path to directory that will store the results. '
                        'Default is ./prediction_{datetime}.')

    parser.add_argument('--params_dir',
                        type=str,
                        help='Path to the directory that holds the \'params\' '
                        'folder and all of the compressed parameter weights. '
                        'Default is ../alphafold/data/.')

    parser.add_argument('--compress_output',
                        action='store_true',
                        help='Whether or not to compress the results '
                        'dictionary that is generated for each query. Default '
                        'is False.')

    parser.add_argument('--dont_write_pdbs',
                        action='store_true',
                        help='Whether or not to write output pdb files. '
                        'Default is False.')

    # Sequence Control Arguments
    parser.add_argument('--min_length',
                        default=1,
                        type=int,
                        help='Minimum single chain length for an AF2 query. '
                        'Default is 1 residue.')

    parser.add_argument('--max_length',
                        default=2500,
                        type=int,
                        help='Maximum single chain length for an AF2 query. '
                        'Default is 2500 residues. If you\'ve got the '
                        'resources and need longer proteins, change this '
                        'argument.')

    parser.add_argument('--max_multimer_length',
                        default=2500,
                        type=int,
                        help='Maximum total protein multimer sequence length '
                        'for an AF2 query. Default is 2500 residues. Note that '
                        'results from AlphaFold-Multimer have not been fully '
                        'validated for multimers > 1536 residues. If you\'ve '
                        'got the resources and need longer protein multimers, '
                        'change this argument.')

    # MSA Arguments
    parser.add_argument('--msa_mode',
                        default='MMseqs2-U+E',
                        type=str,
                        choices=['MMseqs2-U+E', 'MMseqs2-U', 'single_sequence'],
                        help='Mode by which to compute MSA. If .a3m files are '
                        'provided, then AF2 will use the contents of the .a3m '
                        'for the corresponding MSA. Default is MMseqs2-U+E. '
                        'MMseqs2-U+E = Use MMseqs2 to query UniRef and '
                        'Environmental databases. MMseqs2-U = Use MMseqs2 to '
                        'query UniRef database. single_sequence = Don\'t '
                        'generate an MSA.')

    parser.add_argument('--is_prokaryote',
                        action='store_true',
                        help='Whether or not the queries are prokaryotic '
                        'sequences or not. If the origins are eukaryotic or '
                        'unknown, do not include this flag. CURRENTLY DOES NOT '
                        'DO ANYTHING!.')

    # Relaxation Arguments
    parser.add_argument('--use_amber',
                        action='store_true',
                        help='Whether or not to run Amber relaxation. Adding '
                        'this step will increase runtime. By not having this '
                        'step, your models may have small stereochemical '
                        'violations. Default is False.')

    # Template Arguments
    parser.add_argument('--use_templates',
                         action='store_true',
                         help='Whether or not to use templates as determined '
                         'by MMseqs2. Default is False.')

    parser.add_argument('--custom_templates_path',
                         type=str,
                         help='Path to directory containing custom pdb files '
                         'to be used as templates for AF2. Note that '
                         'specifying this will cause the custom templates to '
                         'be used for every query sequence!')

    # Model Control Arguments
    parser.add_argument('--use_ptm',
                         action='store_true',
                         help='Uses the pTM fine-tuned model parameters to '
                         'get PAE per structure. Disable to use the original '
                         'model parameters. Default is False')

    parser.add_argument('--num_models',
                         default=5,
                         type=int,
                         choices=[1, 2, 3, 4, 5],
                         help='Number of models to run. Choose an integer from '
                         '1 to 5. Default is 5.')

    parser.add_argument('--num_ensemble',
                         default=1,
                         type=int,
                         help='The trunk of the network is run multiple times '
                         'with different random choices for the MSA cluster '
                         'centers. Default is 1 but CASP14 settings is 8.')

    parser.add_argument('--random_seed',
                         type=int,
                         help='Random seed for stochastic features of the AF2.')

    parser.add_argument('--num_seeds',
                         default=1,
                         type=int,
                         help='How many different random seeds to try for each '
                         'model and sequence. If --random_seed argument is '
                         'also provided, then it will guarantee that that seed '
                         'will be run. Default is 1.')

    parser.add_argument('--is_training',
                         action='store_true',
                         help='Enables the stochastic part of the model '
                         '(dropout). When coupled with \'num_seeds\' can be '
                         'used to "sample" a diverse set of structures. False '
                         '(NOT including this option) is recommended at first.')

    parser.add_argument('--max_recycle',
                         default=3,
                         type=int,
                         help='Controls the maximum number of times the '
                         'structure is fed back into the neural network for '
                         'refinement. Default is 3.')

    parser.add_argument('--recycle_tol',
                         default=0,
                         type=float,
                         help='Tolerance for deciding when to stop recycling '
                         'the structure through the network (Ca-RMS between '
                         'recycles).')
                        
    return parser
    

class QueryManager(object):
    """Manager that will parse, validate, and store queries. """

    def __init__(self,
            input_dir: str,
            min_length: int = 16,
            max_length: int = 2500,
            max_multimer_length: int = 2500) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.max_multimer_length = max_multimer_length

        self.files = {}
        self.others = []
        self.monomer_queries = []
        self.multimer_queries = []

        # Detect .fasta and .csv files from the input directory.
        onlyfiles = [f for f in os.listdir(input_dir) if os.path.isfile(
                     os.path.join(input_dir, f))]

        for filename in onlyfiles:
            extension = filename.split('.')[-1]
            if extension in ['fasta', 'csv']:
                if extension not in self.files:
                    self.files[extension] = []
                self.files[extension].append(os.path.join(input_dir, filename))
            else:
                self.others.append(os.path.join(input_dir, filename))

        if len(self.files) == 0:
            raise ValueError(
                f'No input .fasta or .csv files detected in '
                '{input_dir}')

        
    def parse_files(self) -> None:

        # For each detected filetype parse queries
        for extension in self.files:
            if extension == 'fasta':
                queries = query_utils.parse_fasta_files(
                    files=self.files['fasta'])
            else:
                queries = query_utils.parse_csv_files(
                    files=self.files['csv'])

            # Validate queries by checking sequence composition and lengths
            queries = query_utils.validate_queries(
                input_queries=queries,
                min_length=self.min_length,
                max_length=self.max_length,
                max_multimer_length=self.max_multimer_length)

            # Add queries to appropriate lists. Important for handling multiple
            # models.
            for query in queries:
                if len(query) == 2:
                    self.monomer_queries.append(query)
                else:
                    self.multimer_queries.append(query)

        # Remove duplicate queries to reduce redundancy
        self.monomer_queries = query_utils.detect_duplicate_queries(
            query_list=self.monomer_queries)
        self.multimer_queries = query_utils.detect_duplicate_queries(
            query_list=self.multimer_queries)        
