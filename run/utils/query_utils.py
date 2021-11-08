""" Utility functions for parsing queries from a list of input files. """

import os
from typing import Sequence, Tuple, Union
import pandas as pd
from alphafold.common import residue_constants

# (filename, sequence)
MonomerQuery = Tuple[str, str]

# (filename, oligomer_state, [sequences])
MultimerQuery = Tuple[str, str, Sequence[str]]


def parse_fasta_files(self, files: Sequence[str]) -> Sequence[MonomerQuery]:
    """ Parse a list of .fasta files and return a list of monomer queries. """
    query_list = []
    
    for filename in files:
        with open(filename, 'r') as f:
            fasta_string = f.read()

        seqs, _ = pipeline.parsers.parse_fasta(fasta_string)

        # .fasta files can contain multiple query sequences.
        for seq in seqs:
            query_list.append( (filename, seq) )

    return query_list


def parse_a3m_files(self, files: Sequence[str]) -> Sequence[MonomerQuery]:
    """ Parse a list of .a3m files and return a list of monomer queries. """
    query_list = []

    for filename in files:
        with open(filename, 'r') as f:
            a3m_string = f.read()

        # Capture the first sequence as the query sequence.
        capture_sequence = False
        for line in a3m_string.splitlines():
            line = line.strip()
            if line.startswith('>'):
                capture_sequence = True # Found first description.
                continue
            elif not line:
                continue # Skip blank lines.
            if capture_sequence:
                sequence = line
                break

        query_list.append( (filename, sequence) )

    return query_list


def parse_csv_files(self, files: Sequence[str]) -> Sequence[MultimerQuery]:
    """ Parse a list of .csv files and return a list of multimer queries. """
    query_list = []

    for filename in files:
        query_df = pd.read_csv(filename, header=None)

        # Each row is a single query with possibly many sequences summarized by
        # the oligomer state.
        for row_idx in range(len(df)):
            oligomer = query_df.iloc[row][0]
            sequences = list(query_df.iloc[row][1:])

            query_list.append( (filename, oligomer, sequences) )

    return query_list


def validate_queries(
        input_queries: Union[Sequence[MonomerQuery], Sequence[MultimerQuery]],
        min_length: int,
        max_length: int,
        max_multimer_length: int
    ) -> Union[Sequence[MonomerQuery], Sequence[MultimerQuery]]:
    """ Validates and cleans input queries. """
    query_list = []

    for query in input_queries:
        if len(query) == 2:
            query = clean_and_validate_monomer_query(
                monomer_query=query,
                min_length=min_length,
                max_length=max_length)
            query_list.append(query)
        else:
            query = clean_and_validate_multimer_query(
                multimer_query=query,
                min_length=min_length,
                max_length=max_length,
                max_multimer_length=max_multimer_length)
            query_list.append(query)

    if len(query_list) > 0:
        return query_list
    
    else:
        raise ValueError('No files contain a valid query, please provide at '
                         'least one query.')

def clean_and_validate_monomer_query(
        monomer_query: MonomerQuery, min_length: int,
        max_length: int) -> MonomerQuery:
    """Checks that the parsed query is ok and returns a clean version of it."""
    filename = monomer_query[0]
    sequence = monomer_query[1]
   
    # Remove all whitespaces, tabs, and end lines; upper-case.
    clean_sequence = sequence.translate(
        str.maketrans('', '', ' \n\t')).upper()
    aatypes = set(residue_constants.restypes) # 20 canonical aatypes.
    if not set(clean_sequence).issubset(aatypes):
        raise ValueError(
            f'Query parsed from {filename} has a sequence with non-amino '
            f'acid letters: {set(clean_sequence) - aatypes}. AlphaFold only '
            f'supports 20 standard amino acids as inputs.')
    if len(clean_sequence) < min_length:
        raise ValueError(
            f'Query parsed from {filename} has a sequence that is too '
            f'short: {len(clean_sequence)} amino acids, while the minimum is '
            f'{min_length}.')
    if len(clean_sequence) > max_length:
        raise ValueError(
            f'Query parsed from {filename} has a sequence that is too '
            f'long: {len(clean_sequence)} amino_acids, while the maximum is '
            f'{max_length}. If you believe you have the resources for this '
            f'query, overwrite the default max_length by providing the '
            f'argument: --max_length NEW_MAX')
    return (filename, clean_sequence)

def clean_and_validate_multimer_query(
        multimer_query: MultimerQuery, min_length: int,
        max_length: int, max_multimer_length: int
        ) -> Union[MultimerQuery, MonomerQuery]:
    """Checks that the parsed query is ok and returns a clean version of it. 
       Also checks if multimer query is actually a monomer query. """
    filename = multimer_query[0]                                                
    oligomer = multimer_query[1]
    sequences = multimer_query[2]

    # Remove whitespaces, tabs, and end lines and upper-case all sequences.
    clean_sequences = []

    for sequence in sequences:
        if pd.isnull(sequence):
            continue
        
        clean_sequence = sequence.translate(                                  
            str.maketrans('', '', ' \n\t')).upper()
        aatypes = set(residue_constants.restypes) # 20 canonical aatypes.
        if not set(clean_sequence).issubset(aatypes):
            raise ValueError(
                f'Query parsed from {filename} has a sequence with '
                f'non-amino acid letters: {set(clean_sequence) - aatypes}. '
                f'AlphaFold only supports 20 standard amino acids as inputs.')
        if len(clean_sequence) < min_length:
            raise ValueError(
                f'Query parsed from {filename} has a sequence that is '
                f'too short: {len(clean_sequence)} amino acids, while the '
                f'minimum is {min_length}.')
        if len(clean_sequence) > max_length:
            raise ValueError(
                f'Query parsed from {filename} has a sequence that is '
                f'too long: {len(clean_sequence)} amino_acids, while the '
                f'maximum is {max_length}. If you believe you have the '
                f'resources for this query, overwrite the default max_length '
                f'by providing the argument: --max_length NEW_MAX.')
        clean_sequences.append(clean_sequence)

    if len(clean_sequences) < 1:
        raise ValueError(
            f'Query parsed from {filename} does not have any detectable '
            f'sequences.')
        
    # Clean oligomer and validate shape
    if pd.isnull(oligomer):
        print(f'WARNING: Inferring oligomeric state from sequences provided in '
              f'{filename}.')
        clean_oligomer = ':'.join(['1'] * len(clean_sequences))

    else:
        clean_oligomer = oligomer.translate(
            str.maketrans('', '', ' \n\t'))
        oligomer_vals = set('123456789:')
        if not set(clean_oligomer).issubset(oligomer_vals):
            raise ValueError(
                f'Query parsed from {filename} has an oligomer state '
                f'with non-valid characters: '
                f'{set(clean_oligomer) - oligomer_vals}.')

        oligos = clean_oligomer.split(':')
        if len(oligos) > len(clean_sequences):
            raise ValueError(
                f'Query parsed from {filename} has more oligomeric '
                f'states than number of sequences: oligomer = '
                f'{clean_oligomer}, num_seqs = {len(clean_sequences)}.')
        if len(oligos) < len(clean_sequences):
            raise ValueError(
                f'Query parsed from {filename} has less oligomeric '
                f'states than number of sequences: oligomer = '
                f'{clean_oligomer}, num_seqs = {len(clean_sequences)}.')

    total_multimer_length = sum(
        [len(seq) * int(oligo) for seq, oligo in zip(clean_sequences, oligos)])

    if total_multimer_length > max_multimer_length:
        raise ValueError(
            f'Query parsed from {filename} has a total length of the '
            f'multimer that is too long: {total_multimer_length}, while the '
            f'maximum is {max_multimer_length}. If you believe you have the '
            f'resources for this query, overwrite the default '
            f'max_multimer_length by providing the argument: '
            f'--max_multimer_length NEW_MAX.')
    elif total_multimer_length > 1536:
        print(f'WARNING: The accuracy of the system has not been fully '
              f'validated above 1536 residues. Query from {filename} is '
              f'a total length of {total_multimer_length}.')    

    if len(clean_sequences) == 1 and
        len(total_multimer_length) == len(clean_sequences[0]):

        return (filename, clean_sequences[0])
    else:
        return (filename, clean_oligomer, clean_sequences)

def detect_duplicate_queries(
        query_list: Union[Sequence[MonomerQuery], Sequence[MultimerQuery]]
        ) -> query_list: Union[Sequence[MonomerQuery], Sequence[MultimerQuery]]:
    """ Detects duplicate queries from query list. If a same query comes from 
        two different sources, it is considered a duplicate unless one of those
        sources is an .a3m file. .a3m files are always considered unique due to 
        the potential for custom MSAs. """
    clean_query_list = []

    for query in query_list:
        if len(clean_query_list) == 0:
            clean_query_list.append(query)
        else:
            dupe = False
            
            for old_query in clean_query_list:
                for idx, aspect in enumerate(query):
                    if idx == 0:
                        if aspect[-4:] == '.a3m':
                            break
                    else:
                        if aspect == old_query[idx]:
                            if old_query[0][-4:] != '.a3m':
                                dupe = True
                                break

            if dupe == False:
                clean_query_list.append(query)

    return clean_query_list


def getFullSequence(
        query: Union[MonomerQuery, MultimerQuery]
        ) -> str:

    if len(query) == 2:
        full_sequence = query[1]

    else:
        oligomer = query[1]
        sequences = query[2]

        oligo_list = oligomer.split(':')

        full_sequence = ''.join([
            seq * int(oligo) for seq, oligo in zip(sequences, oligo_list)])

    return full_sequence

    
