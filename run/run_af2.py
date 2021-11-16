""" Full AlphaFold protein structure prediction script."""

# Standard imports.
import os
import sys
import logging
import time
import jax

# Update PATH.
sys.path.append('~/anaconda3/envs/af2/lib/python3.7/site-packages')
sys.path.append('./content/alphafold')

# AlphaFold imports.
from alphafold.common import protein
from alphafold.relax import relax

# Custom imports.
from setup import getAF2Parser, QueryManager, getOutputDir
from features import getRawInputs, getChainFeatures, getInputFeatures
from model import (
    getRandomSeeds, getModelNames, getModelRunner, predictStructure)
from utils.query_utils import getFullSequence
from utils.utils import compressed_pickle, get_hash, full_pickle

# Parse arguments.
parser = getAF2Parser()
args = parser.parse_args()
del parser

# Update output directory.
output_dir = getOutputDir(out_dir=args.output_dir)

# Set up logger
logging.basicConfig(filename=os.path.join(output_dir, 'prediction.log'),
                    level=logging.INFO)
logger = logging.getLogger('run_af2')

# Log what device program is running on.
logger.info(f'Running with {jax.local_devices()[0].device_kind} '
            f'{jax.local_devices()[0].platform.upper()}')

# Update environmental variables.
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

# Set up timing dictionary.
timings = {}
t_all = time.time()

# Parse queries.
qm = QueryManager(
    input_dir=args.input_dir,
    min_length=args.min_length,
    max_length=args.max_length,
    max_multimer_length=args.max_multimer_length)
qm.parse_files()

queries = qm.queries
logger.info(f'Queries have been parsed. {len(queries)} queries found.')
del qm

# Get raw model inputs.
t_0 = time.time()
raw_inputs_from_sequence = getRawInputs(
    queries=queries,
    msa_mode=args.msa_mode,
    use_templates=args.use_templates,
    output_dir=output_dir)
timings['raw_inputs'] = time.time() - t_0
logger.info(f'Raw inputs have been generated. Took {timings["raw_inputs"]:.2f} '
            f'seconds.')

# Get random seeds.
seeds = getRandomSeeds(
    random_seed=args.random_seed,
    num_seeds=args.num_seeds)

# Get model names.
model_names = getModelNames(
    first_n_seqs=len(queries[0][1]),
    last_n_seqs=len(queries[-1][1]),
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
    logger.info(f'Obtained model runner for {model_name}.')

    file_id = None
    for query_idx, query in enumerate(queries):
        # Skip any multimer queries if current model_runner is a monomer model.
        if len(query[1]) > 1 and 'multimer' not in model_name:
            continue
        # Skip any monomer queries if current model_runner is a multimer model.
        elif len(query[1]) == 1 and 'multimer' in model_name:
            continue

        if file_id == None:
            file_id = query[0]
            idx = 0
        elif file_id == query[0]:
            idx += 1
        else:
            file_id = query[0]
            idx = 0

        prefix = '.'.join(file_id.split('.')[:-1]) + f'_{idx}_' + model_name
        sequences = query[1]

        t_0 = time.time()
        features_for_chain = getChainFeatures(
            sequences=sequences,
            raw_inputs=raw_inputs_from_sequence,
            use_templates=args.use_templates,
            custom_msa_path=args.custom_msa_path,
            custom_template_path=args.custom_template_path)
        
        input_features = getInputFeatures(
            sequences=sequences,
            chain_features=features_for_chain,
            is_prokaryote=args.is_prokaryote)
        timings[f'features_{model_name}_{query_idx}'] = time.time() - t_0
        logger.info(f'Features for {model_name}, query {query_idx} have been '
                    f'generated. Took '
                    f'{timings[f"features_{model_name}_{query_idx}"]} '
                    f'seconds.')
        
        del sequences, features_for_chain

        for seed_idx, seed in enumerate(seeds):
            if 'multimer' in model_name:
                model_type = 'multimer'
            else:
                model_type = 'monomer'

            jobname = prefix + f'_{seed_idx}'

            t_0 = time.time()
            result = predictStructure(
                model_runner=model_runner,
                feature_dict=input_features,
                model_type=model_type,
                random_seed=seed)
            timings[f'predict_{model_name}_{seed_idx}_{query_idx}'] = time.time() - t_0
            logger.info(f'Structure prediction for {model_name}, seed '
                        f'{seed_idx}, query {query_idx} is completed. Took '
                        f'{timings[f"predict_{model_name}_{seed_idx}_{query_idx}"]} '
                        f'seconds.')
            
            if not args.dont_write_pdbs:
                unrelaxed_pdb = protein.to_pdb(result['unrelaxed_protein'])

                unrelaxed_pred_path = os.path.join(
                    output_dir, f'{jobname}_unrelaxed.pdb')
                with open(unrelaxed_pred_path, 'w') as f:
                    f.write(unrelaxed_pdb)
                logger.info('Unrelaxed protein pdb has been written.')
                    
                del unrelaxed_pdb, unrelaxed_pred_path
            
            if args.use_amber:
                t_1 = time.time()
                relaxed_pdb, _, _ = amber_relaxer.process(
                    prot=result['unrelaxed_protein'])
                timings[f'relax_{model_name}_{seed_idx}_{query_idx}'] = time.time() - t_1
                logger.info('Protein structure has been relaxed with AMBER.')
                
                if not args.dont_write_pdbs:
                    relaxed_pred_path = os.path.join(
                        output_dir, f'{jobname}_relaxed.pdb')
                    with open(relaxed_pred_path, 'w') as f:
                        f.write(relaxed_pdb)
                    logger.info('Relaxed protein pdb has been written.')
                        
                    del relaxed_pred_path

                del t_1, relaxed_pdb
                
            results_path = os.path.join(
                output_dir, f'{jobname}_results')
            if args.compress_output:
                compressed_pickle(results_path, result)
                logger.info('Results have been pickled and compressed.')
            else:
                full_pickle(results_path, result)
                logger.info('Results have been pickled.')
    
            del model_type, jobname, result, results_path

timings['overall'] = time.time() - t_all
logger.info(f'Overall prediction process took {timings["overall"]} seconds.')

if args.save_timing:
    timing_path = os.path.join(output_dir, 'timing')
    if args.compress_output:
        compressed_pickle(timing_path, timings)
    else:
        full_pickle(timing_path, timings)
