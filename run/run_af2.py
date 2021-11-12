""" Full AlphaFold protein structure prediction script."""

import jax
print(f'Running with {jax.local_devices()[0].device_kind} '
      '{jax.local_devices()[0].platform.upper()}')

import sys
sys.path.append('~/anaconda3/envs/af2/lib/python3.7/site-packages')
sys.path.append('./content/alphafold')

import os
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

from setup import getAF2Parser, QueryManager
from features import getRawInputs
from features import getChainFeatures, getInputFeatures
from model import getRandomSeeds, getModelNames, getModelRunner
from model import predictStructure
from utils.query_utils import getFullSequence
from utils.utils import compressed_pickle, get_hash, full_pickle

from alphafold.common import protein
from alphafold.relax import relax

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

# Parse arguments.
parser = getAF2Parser()
args = parser.parse_args()

# Parse queries.
qm = QueryManager(
    input_dir=args.input_dir,
    min_length=args.min_length,
    max_length=args.max_length,
    max_multimer_length=args.max_multimer_length)
qm.parse_files()

queries = qm.monomer_queries + qm.multimer_queries
del qm

# Get raw model inputs.
raw_inputs_from_sequence = getRawInputs(
    queries=queries,
    msa_mode=args.msa_mode,
    use_templates=args.use_templates,
    output_dir=args.output_dir)

# Get random seeds.
seeds = getRandomSeeds(
    random_seed=args.random_seed,
    num_seeds=args.num_seeds)

# Get model names.
model_names = getModelNames(
    first_query_len=len(queries[0]),
    last_query_len=len(queries[-1]),
    use_ptm=args.use_ptm, num_models=args.num_models)

if args.use_amber:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0, tolerance=2.39, stiffness=10.0,
        exclude_residues=[], max_outer_iterations=3)
    
# Predict structures.
for model_name in model_names:

    model_runner = getModelRunner(
        model_name=model_name,
        num_ensemble=args.num_ensemble,
        is_training=args.is_training,
        num_recycle=args.max_recycle,
        recycle_tol=args.recycle_tol,
        params_dir=args.params_dir)

    for query in queries:
        # Skip any multimer queries if current model_runner is a monomer model.
        if len(query) == 3 and 'multimer' not in model_name:
            continue
        # Skip any monomer queries if current model_runner is a multimer model.
        elif len(query) == 2 and 'multimer' in model_name:
            continue

        prefix = query[0].split('.')[:-1] + model_name
        
        sequences = query[-1]

        if isinstance(sequences, str):
            sequences = [sequences]

        features_for_chain = getChainFeatures(
            sequences=sequences,
            raw_inputs=raw_inputs_from_sequence,
            use_templates=args.use_templates,
            custom_a3m_lines=args.custom_msa_path,
            custom_templates_path=args.custom_template_path)
        
        input_features = getInputFeatures(
            sequences=sequences,
            chain_features=features_for_chain,
            is_prokaryote=args.is_prokaryote)
        
        del filename, custom_a3m, sequences, features_for_chain
        del full_sequence

        for seed_idx, seed in enumerate(seeds):
            if 'multimer' in model_name:
                model_type = 'multimer'
            else:
                model_type = 'monomer'

            jobname = prefix + f'seed_{seed_idx}'
                
            result = predictStructure(
                model_runner=model_runner,
                feature_dict=input_features,
                model_type=model_type,
                random_seed=seed)

            if not args.dont_write_pdbs:
                unrelaxed_pdb = protein.to_pdb(result['unrelaxed_protein'])

                unrelaxed_pred_path = os.path.join(
                    args.output_dir, f'{jobname}_unrelaxed.pdb')
                with open(unrelaxed_pred_path, 'w') as f:
                    f.write(unrelaxed_pdb)

                del unrelaxed_pdb, unrelaxed_pred_path
            
            if args.use_amber:
                relaxed_pdb, _, _ = amber_relaxer.process(
                    prot=result['unrelaxed_protein'])

                if not args.dont_write_pdbs:
                    relaxed_pred_path = os.path.join(
                        args.output_dir, f'{jobname}_relaxed.pdb')
                    with open(relaxed_pred_path, 'w') as f:
                        f.write(relaxed_pdb)

                    del relaxed_pred_path

                del relaxed_pdb

            results_path = os.path.join(
                args.output_dir, f'{jobname}_results')
            if args.compress_output:
                compressed_pickle(results_path, result)
            else:
                full_pickle(results_path, result)
                    
            del result, results_path
