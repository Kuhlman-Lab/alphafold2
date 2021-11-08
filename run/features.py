"""
Methods for generating features for AlphaFold. Includes MSA and template 
generation
"""

import os
import requests
import time
import random
import tarfile
import copy
import numpy as np
import pathlib
from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.notebooks import notebook_utils
from typing import Sequence, Optional, Dict, Tuple, MutableMapping


# (filename, sequence)
MonomerQuery = Tuple[str, str]

# (filename, oligomer_state, [sequences])
MultimerQuery = Tuple[str, str, Sequence[str]]

# {sequence: (raw MSA, raw templates)}
RawInput = Dict[str, Tuple[str, str]]

def getMonomerRawInputs(
        monomer_queries: Sequence[MonomerQuery],
        use_env: bool = True,
        use_filter: bool = True,
        use_templates: bool = False,
        output_dir: str = '',
        raw_inputs: Optional[RawInput] = None,
        ) -> Tuple(RawInput, Dict[str, str]):
    """ Computes and gathers the raw MSAs for the list of monomer queries. """
    # Get already computed MSAs and templates.
    if raw_inputs_from_sequence:
        raw_inputs = raw_inputs_from_sequence
    else:
        raw_inputs = {}
    
    # Gather monomer sequences to run MMseqs2 in a batch.
    new_sequences = []
    a3m_files = []
    for monomer in monomer_queries:
        filename = monomer[0]
        seq = monomer[1]

        # Don't include .a3m files as they already have a custom MSA.
        if filename[-4:] != '.a3m':
            # Don't do redundant work if chain already has MSA and templates.
            if seq not in raw_inputs:
                new_sequences.append(seq)
        else:
            a3m_files.append(filename)

    # Run MMseqs2.
    a3m_lines, template_paths = runMMseqs2(
        prefix=os.path.join(output_dir, 'mmseq2', 'monomers')
        sequences=new_sequences,
        use_env=use_env,
        use_filter=use_filter,
        use_templates=use_templates)

    # Store into dictionary.
    for msa, templates in zip(a3m_lines, template_paths):
        sequence = msa.splitlines()[1]
        raw_inputs[sequence] = (msa, templates)

    # Parse any .a3m files.
    a3ms = {}
    for a3m in a3m_files:
        with open(a3m, 'r') as f:
            a3ms[a3m] = f.read()

    return (raw_inputs, a3ms)


def getMultimerRawInputs(
        multimer_queries: Sequence[MultimerQuery],
        use_env: bool = True,
        use_filter: bool = True,
        use_templates: bool = False,
        output_dir: str = '',
        raw_inputs: Optional[RawInput] = None,
        ) -> RawInput:
    # Get already computed MSAs and templates.
    if raw_inputs_from_sequence:
        raw_inputs = raw_inputs_from_sequence
    else:
        raw_inputs = {}

    # Gather multimer sequences to run MMseqs in a batch.
    new_sequences = []
    for multimer in multimer_queries:
        filename = multimer[0]
        oligomer = multimer[1]
        sequences = multimer[2]

        for sequence in sequences:
            if sequence not in raw_inputs:
                new_sequences.append(sequence)

    # Run MMseqs2.
    a3m_lines, template_paths = runMMseqs2(
        prefix=os.path.join(output_dir, 'mmseqs2', 'multimers')
        sequences=new_sequences,
        use_env=use_env,
        use_filter=use_filter,
        use_templates=use_templates)

    # Store into dictionary.
    for msa, templates in zip(a3m_lines, template_paths):
        sequence = msa.splitlines()[1]
        raw_inputs[sequence] = (msa, templates)

    return raw_inputs


def runMMseqs2(
        prefix: str,
        sequences: Union[Sequence[str], str],
        use_env: bool = True,
        use_filter: bool = True,
        use_templates: bool = False,
        num_templates: int = 20,
        host_url: str = 'https:/a3m.mmseqs.com'
        ) -> Tuple[Sequence[str], Sequence[Optional[str]]]:
    """ Computes MSAs and templates by querying MMseqs2 API. """

    def submit(
            seqs: Sequence[str],
            mode: str,
            N: int) -> Dict[str, str]:
        """ Submits a query of sequences to MMseqs2 API. """

        # Make query from list of sequences.
        n, query = N, ''
        for seq in seqs:
            query += f'>{n}\n{seq}\n'
            n += 1

        res = requests.post(f'{host_url}/ticket/msa',
                            data={'q': query, 'mode': mode})
        try:
            out = res.json()
        except ValueError:
            out = {'status': 'UNKNOWN'}

        return out

    def status(ID: int) -> Dict[str, str]:
        """ Obtains the status of a submitted query. """
        res = requests.get(f'{host_url}/ticket/{ID}')
        try:
            out = res.json()
        except ValueError:
            out = {'status': 'UNKNOWN'}

        return out

    def download(ID: int, path: str) -> None:
        """ Downloads the completed MMseqs2 query. """
        res = requests.get(f'{host_url}/result/download/{ID}')
        with open(path, 'wb') as out:
            out.write(res.content)
    
    # Make input sequence a list if not already.
    sequences = [sequences] if isinstance(sequences, str) else sequences

    # Set the mode for MMseqs2.
    if use_filter:
        mode = 'env' if use_env else 'all'
    else:
        mode = 'env-nofilter' if use_env else 'nofilter'

    # Set up output path.
    out_path = f'{prefix}_{mode}'
    os.makedirs(out_path, exist_ok=True)
    tar_gz_file = os.path.join(out_path, 'out.tar.gz')
    N, REDO = 101, True

    # Deduplicate and keep track of order.
    unique_seqs = sorted(list(set(seqs)))
    Ms = [N + unique_seqs.index(seq) for seq in seqs]

    # Call MMseqs2 API.
    if not os.path.isfile(tar_gz_file):
        while REDO:
            # Resubmit job until it goes through
            out = submit(seqs=seqs_unique, mode=mode, N=N)
            while out['status'] in ['UNKNOWN', 'RATELIMIT']:
                # Resubmit
                time.sleep(5 + random.randint(0, 5))
                out = submit(seqs=seqs_unique, mode=mode, N=N)

            # Wait for job to finish
            ID = out['id']
            while out['status'] in ['UNKNOWN', 'RUNNING', 'PENDING']:
                time.sleep(5 + random.randint(0, 5))
                out = status(ID)

            if out['status'] == 'COMPLETE':
                REDO = False
            elif out['status'] == 'ERROR':
                REDO = False
                raise Exception('MMseqs2 API is giving errors. Please confirm '
                                'your input is a valid protein sequence. If '
                                'error persists, please try again in an hour.')
            # Download results
            download(ID, tar_gz_file)

    # Get and extract a list of .a3m files.
    a3m_files = [os.path.join(out_path, 'uniref.a3m')]
    if use_env:
        a3m_files.append(
            os.path.join(out_path, 'bfd.mgnify30.metaeuk30.smag30.a3m'))
    if not os.path.isfile(a3m_files[0]):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # Get templates if necessary.
    if use_templates:
        templates = {}
        
        # Read MMseqs2 template outputs and sort templates based on query seq.
        with open(os.path.join(out_path, 'pdb70.m8', 'r')) as f:
            for line in f:
                p = line.rstrip().split()
                M, pdb = p[0], p[1]
                M = int(M)
                if M not in templates:
                    templates[M] = []
                templates[M].append(pdb)

        # Obtain template structures and data files
        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = os.path.join(prefix+'_'+mode, f'templates_{k}')
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ','.join(TMPL[:num_templates])
                # Obtain the .cif and data files for the templates
                os.system(
                    f'curl -s '
                    f'https://a3m-templates.mmseqs.com/template/{TMPL_LINE} '
                    f'| tar xzf - -C {TMPL_PATH}/')
                # Rename data files
                os.system(
                    f'cp {TMPL_PATH}/pdb70_a3m.ffindex '
                    f'{TMPL_PATH}/pdb70_cs219.ffindex')
                os.system(f'touch {TMPL_PATH}/pdb70_cs219.ffdata')
            template_paths[k] = TMPL_PATH

    # Gather .a3m lines.
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file, 'r') as f:
            for line in f:
                if len(line) > 0:
                    # Replace NULL values
                    if '\x00' in line:
                        line = line.replace('\x00', '')
                        update_M = True
                    if line.startswith('>') and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    # Return results.
    a3m_lines = [''.join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_
    else:
        template_paths = []
        for n in Ms:
            template_paths.append(None)

    if isinstance(sequences, str):
        return (a3m_lines[0], template_paths[0])
    else:
        return (a3m_lines, template_paths)
            

def getMSA(
        sequence: str,
        raw_inputs_from_sequence: Optional[RawInput] = None
        custom_a3m_lines: Optional[str] = None) -> parsers.Msa:

    # Get single-chain MSA.
    if custom_a3m_lines:
        a3m_lines = custom_a3m_lines
    elif raw_inputs_from_sequence:
        raw_inputs = copy.deepcopy(raw_inputs_from_sequence[sequence])
        a3m_lines = raw_inputs[0]
        template_paths = raw_inputs[1]

    single_chain_msa = [parsers.parse_a3m(a3m_string=a3m_lines)]

    # Uniprot MSA?
    uniprot_msas = []
        
    return single_chain_msa


def getChainFeatures(
        sequences: Sequence[str],
        raw_inputs: RawInput,
        use_templates: bool = False,
        custom_a3m_lines: Optional[str] = None
        ) -> MutableMapping[str, pipeline.FeatureDict]:
    feature_for_chain = {}
    
    for sequence_idx, sequence in enumerate(sequences):
        feature_dict = {}
        # Get sequence features
        feature_dict.update(pipeline.make_sequence_features(
            sequence=sequence, description='query', num_res=len(sequence)))

        # Get MSA features
        msa = getMSA(
            sequence=sequence, raw_inputs=raw_inputs_from_sequence,
            custom_a3m_lines=custom_a3m_lines)
        feature_dict.update(pipeline.make_msa_features(msas=msa))

        # Get template features
        if use_templates:
            new_raw_inputs = copy.deepcopy(raw_inputs[sequence])
            a3m = new_raw_inputs[0]
            template = new_raw_inputs[1]
            
            feature_dict.update(
                make_template(sequence, a3m, template))
        else:
            feature_dict.update(
                notebook_utils.empty_placeholder_template_features(
                    num_templates=0, num_res=len(sequence)))

        features_for_chain[
            protein.PDB_CHAIN_IDS[sequence_idx - 1]] = feature_dict
        
    return feature_for_chain


def getInputFeatures(
        sequences: Sequence[str],
        chain_features: MutableMapping[str, pipeline.FeatureDict],
        is_prokaryote: bool = False,
        min_num_seq: int = 512
        ) -> Union[pipeline.FeatureDict,
                   MutableMapping[str, pipeline.FeatureDict]]:

    if len(sequences) == 1:
        return chain_features[protein.PDB_CHAIN_IDS[0]]
    else:
        all_chain_features = {}
        for chain_id, features in chain_features.items():
            all_chain_features[
                chain_id] = pipeline_multimer.convert_monomer_features(
                    features, chain_id)

            all_chain_features = pipeline_multimer.add_assembly_features(
                all_chain_features)

            input_features = feature_processing.pair_and_merge(
                all_chain_features=all_chain_features,
                is_prokaryote=is_prokaryote)

            # Pad MSA to avoid zero-size extra MSA.
            return pipeline_multimer.pad_msa(input_features,
                                             min_num_seq=min_num_seq)


def make_template(
        query_sequence: str,
        a3m_lines: Sequence[str],
        template_paths: str):

    template_featurizer = template_utils.TemplateHitFeaturizer(
            mmcif_dir=template_paths,
            max_template_date='2100-01-01',
            max_hits=20,
            kalign_binary_path='kalign',
            release_dates_path=None,
            obsolete_pdbs_path=None)

    hhsearch_pdb70_runner = template_utils.HHSearch(
        binary_path='hhsearch',
        databases=[f'{template_paths}/pdb70'])

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hits=hhsearch_hits)

    return templates_result.features
