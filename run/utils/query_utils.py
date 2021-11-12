""" Utility functions for parsing queries from a list of input files. """

import os
import csv
from typing import Sequence, Tuple, Union
from alphafold.common import residue_constants
from alphafold.data import pipeline

# (filename, sequence)
MonomerQuery = Tuple[str, str]

# (filename, oligomer_state, [sequences])
MultimerQuery = Tuple[str, str, Sequence[str]]


def parse_fasta_files(files: Sequence[str]) -> Sequence[MonomerQuery]:
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


def parse_csv_files(files: Sequence[str]) -> Sequence[MultimerQuery]:
    """ Parse a list of .csv files and return a list of multimer queries. """
    query_list = []

    for filename in files:
        with open(filename, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            # Each row is a single query with possibly many sequences
            # summarized by the oligomer state.
            for row in reader:
                oligomer = row[0]
                sequences = row[1:]

                query_list.append( (filename, oligomer, sequences) )

    return query_list


def clean_and_validate_queries(
        input_queries: Union[Sequence[MonomerQuery], Sequence[MultimerQuery]],
        min_length: int,
        max_length: int,
        max_multimer_length: int
    ) -> Union[Sequence[MonomerQuery], Sequence[MultimerQuery]]:
    """ Validates and cleans input queries. """
    query_list = []

    for query in input_queries:
        query = _clean_and_validate_single_query(
            query=query,
            min_length=min_length,
            max_length=max_length,
            max_multimer_length=max_multimer_length)
        query_list.append(query)
        
    if len(query_list) > 0:
        return query_list
    
    else:
        raise ValueError('No files contain a valid query, please provide at '
                         'least one query.')

    
def _clean_and_validate_single_query(
        query: Union[MonomerQuery, MultimerQuery], min_length: int,
        max_length: int, max_multimer_length: int) -> MonomerQuery:
    """Checks that the parsed query is ok and returns a clean version of it."""
    filename = query[0]
    sequences = query[-1]
    if len(query) == 2:
        # If a monomer query is given, then it has an oligomeric state of 1.
        oligomer = '1'
    else:
        oligomer = query[1]

    # If a monomer query is given, then it has single sequence. Need to treat as
    # a list of sequences.
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Clean filename by removing all parent directories.
    clean_filename = os.path.basename(filename)

    # Remove whitespaces, tabs, and end lines and uppercase all sequences.
    clean_sequences = []
    for sequence in sequences:
        clean_sequence = sequence.translate(
            str.maketrans('', '', ' \n\t')).upper()
        aatypes = set(residue_constants.restypes) # 20 canonical aatypes.
        if not set(clean_sequence).issubset(aatypes):
            raise ValueError(
                f'Query parsed from {clean_filename} has a sequence with '
                f'non-amino acid letters: {set(clean_sequence) - aatypes}. '
                f'AlphaFold only supports 20 standard amino acids as inputs.')
        if len(clean_sequence) < min_length:
            raise ValueError(
                f'Query parsed from {clean_filename} has a sequence that is '
                f'too short: {len(clean_sequence)} amino acids, while the '
                f'minimum is {min_length}.')
        if len(clean_sequence) > max_length:
            raise ValueError(
                f'Query parsed from {clean_filename} has a sequence that is '
                f'too long: {len(clean_sequence)} amino acids, while the '
                f'maximum is {max_length}. If you believe you have the '
                f'resources for this query, overwrite the default max_length '
                f'by providing the argument: --max_length NEW_MAX.')
        clean_sequences.append(clean_sequence)

    if len(clean_sequences) < 1:
        raise ValueError(
            f'Query parsed from {clean_filename} does not have any detectable '
            f'sequences.')

    # Clean oligomer and validate shape
    if oligomer == '':
        print(f'WARNING: Inferring oligomeric state from sequences provided in '
              f'{clean_filename}.')
        clean_oligomer = ':'.join(['1'] * len(clean_sequences))
    else:
        clean_oligomer = oligomer.translate(
            str.maketrans('', '', ' \n\t'))

    oligomer_vals = set('123456789:')
    if not set(clean_oligomer).issubset(oligomer_vals):
        raise ValueError(
            f'Query parsed from {clean_filename} has an oligomer state '
            f'with non-valid characters: '
            f'{set(clean_oligomer) - oligomer_vals}.')
    oligos = clean_oligomer.split(':')
    if len(oligos) > len(clean_sequences):
        raise ValueError(
            f'Query parsed from {clean_filename} has more oligomeric '
            f'states than number of sequences: {len(oligos)} > '
            f'{len(clean_sequences)}. Oligomer is {clean_oligomer}.')
    if len(oligos) < len(clean_sequences):
        raise ValueError(
            f'Query parsed from {clean_filename} has less oligomeric '
            f'states than number of sequences: {len(oligos)} < '
            f'{len(clean_sequences)}. Oligomer is {clean_oligomer}.')

    total_multimer_length = sum(
        [len(seq) * int(oligo) for seq, oligo in zip(clean_sequences, oligos)])
    if total_multimer_length > max_multimer_length:
        raise ValueError(
            f'Query parsed from {clean_filename} has a total multimer length '
            f'that is too long: {total_multimer_length}, while the maximum '
            f'is {max_multimer_length}. If you believe you have the resources '
            f'for this query, overwrite the default max_multimer_length by '
            f'providing the argument: --max_multimer_length NEW_MAX.')
    elif total_multimer_length > 1536:
        print(f'WARNING: The accuracy of the multimer system has not been '
              f'fully validated above 1536 residues. Query from '
              f'{clean_filename} is a total length of {total_multimer_length}.')

    # If there is only one sequence and it is the same length as total multimer
    # then the query is a monomer query.
    if len(clean_sequences) == 1 and (
            total_multimer_length == len(clean_sequences[0])):
        return (clean_filename, clean_sequences[0])
    else:
        return (clean_filename, clean_oligomer, clean_sequences)

    
def detect_duplicate_queries(
        query_list: Sequence[Union[MonomerQuery, MultimerQuery]]
        ) -> Sequence[Union[MonomerQuery, MultimerQuery]]:
    """ Detects duplicate queries from query list. If a same query comes from 
        two different sources, it is considered a duplicate. """
    clean_query_list = []

    for query in query_list:
        # If the clean_query_list is empty, query is not a dupe.
        if len(clean_query_list) == 0:
            clean_query_list.append(query)
        else:
            dupe = False
            
            for old_query in clean_query_list:
                if _check_dupe(old_query, query):
                    dupe = True

            if dupe == False:
                clean_query_list.append(query)

    return clean_query_list


def _check_dupe(old_query: Union[MonomerQuery, MultimerQuery],
                new_query: Union[MonomerQuery, MultimerQuery]) -> bool:

    old_fullseq = getFullSequence(query=old_query)
    new_fullseq = getFullSequence(query=new_query)
    if old_fullseq == new_fullseq:
        return True
    else:
        return False
    

def getFullSequence(query: Union[MonomerQuery, MultimerQuery]) -> str:
    if len(query) == 2:
        full_sequence = query[1]
    else:
        oligomer = query[1]
        sequences = query[2]

        oligo_list = oligomer.split(':')

        full_sequence = ''.join([
            seq * int(oligo) for seq, oligo in zip(sequences, oligo_list)])

    return full_sequence

    
