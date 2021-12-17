""" Full AlphaFold protein structure prediction script."""

# Standard imports.
import os
import sys
import logging
import time
from typing import Sequence, Union, Optional

# Update PATH.
sys.path.append('~/anaconda3/envs/af2/lib/python3.7/site-packages')
sys.path.append('./content/alphafold')

# Custom imports.
from setup import getAF2Parser, QueryManager, getOutputDir
from utils.utils import compressed_pickle, get_hash, full_pickle

# Global constants.
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def af2(sequences: Optional[Sequence[Sequence[str]]] = [],
        arg_file: Optional[str] = None,
        proc_id: Optional[int] = None,
        fitness_fxn = None) -> Optional[Sequence[float]]:

    parser = getAF2Parser()
    if arg_file != None:
        args = parser.parse_args([f'@{arg_file}'])
    else:
        args = parser.parse_args(sys.argv[1:])
    del parser

    output_dir = getOutputDir(out_dir=args.output_dir)
    
    # Set up logger
    if not args.no_logging and not args.design_run:
        logging.basicConfig(filename=os.path.join(output_dir, 'prediction.log'),
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join(output_dir, 'prediction.log'))
    logger = logging.getLogger('run_af2')

    # Update environmental variables.
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
    if proc_id != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(proc_id)
    
    # Get all AF2 imports.
    import jax
    from alphafold.common import protein
    from alphafold.relax import relax
    from features import (
        getRawInputs, getChainFeatures, getInputFeatures)
    from model import (
        getRandomSeeds, getModelNames, getModelRunner, predictStructure)
    from utils.query_utils import getFullSequence

    # Log devices
    devices = jax.local_devices()
    NUM_DEVICES = len(devices)
    logger.info(f'Running JAX with {NUM_DEVICES} devices.')
    
    # Set up timing dictionary.
    timings = {}
    t_all = time.time()
    
    # Parse queries.
    qm = QueryManager(
        input_dir=args.input_dir,
        sequences=sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        max_multimer_length=args.max_multimer_length)
    qm.parse_files()
    qm.parse_sequences()

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
    logger.info(f'Raw inputs have been generated. Took '
                f'{timings["raw_inputs"]:.2f} seconds.')

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
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            stiffness=RELAX_STIFFNESS,
            exclude_residues=RELAX_EXCLUDE_RESIDUES,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

    # Precompute query features
    query_features = []

    file_id = None
    for query_idx, query in enumerate(queries):
        if file_id == None:
            file_id = query[0]
            idx = 0
        elif file_id == query[0]:
            idx += 1
        else:
            file_id = query[0]
            idx = 0

        prefix = '.'.join(file_id.split('.')[:-1]) + f'_{idx}'
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
        timings[f'features_{query_idx}'] = time.time() - t_0
        logger.info(f'Features for query {query_idx} have been generated. Took '
                    f'{timings[f"features_{query_idx}"]} seconds.')

        query_features.append( (prefix, sequences, input_features) )

    results_list = []

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

        if 'multimer' in model_name:
            run_multimer = True
        else:
            run_multimer = False

        for query_idx, query in enumerate(query_features):
            prefix = query[0] + f'_{model_name}'
            sequences = query[1]

            if len(sequences) > 1 and 'multimer' not in model_name:
                continue
            elif len(sequences) == 1 and 'multimer' in model_name:
                continue
            
            input_features = query[2]

            del sequences

            for seed_idx, seed in enumerate(seeds):
                jobname = prefix + f'_{seed_idx}'

                t_0 = time.time()
                result = predictStructure(
                    model_runner=model_runner,
                    feature_dict=input_features,
                    run_multimer=run_multimer,
                    random_seed=seed)
                results_list.append(result)
                timings[f'predict_{model_name}_{seed_idx}'] = (time.time() - t_0)
                logger.info(f'Structure prediction for {model_name}, seed '
                            f'{seed_idx} is completed. Took '
                            f'{timings[f"predict_{model_name}_{seed_idx}"]} '
                            f'seconds.')
            
                if not args.dont_write_pdbs and not args.design_run:
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
                    timings[f'relax_{model_name}_{seed_idx}_{query_idx}'] = (
                        time.time() - t_1)
                    logger.info('Structure has been relaxed with AMBER.')
                
                    if not args.dont_write_pdbs and not args.design_run:
                        relaxed_pred_path = os.path.join(
                            output_dir, f'{jobname}_relaxed.pdb')
                        with open(relaxed_pred_path, 'w') as f:
                            f.write(relaxed_pdb)
                        logger.info('Relaxed protein pdb has been written.')
                        
                        del relaxed_pred_path

                    del t_1, relaxed_pdb

                if not args.design_run:
                    results_path = os.path.join(output_dir, f'{jobname}_results')
                    if args.compress_output:
                        compressed_pickle(results_path, result)
                        logger.info('Results have been pickled and compressed.')
                    else:
                        full_pickle(results_path, result)
                        logger.info('Results have been pickled.')
                    del results_path
                        
                del jobname, result, prefix, input_features
            # end for seed in seeds
            
        # end for query in queries
        del run_multimer
        
    # end for model_name in model_names
        
    timings['overall'] = time.time() - t_all
    logger.info(f'Overall prediction process took {timings["overall"]} '
                f'seconds.')

    if args.save_timing:
        timing_path = os.path.join(output_dir, 'timing')
        if args.compress_output:
            compressed_pickle(timing_path, timings)
        else:
            full_pickle(timing_path, timings)

    if fitness_fxn:
        fitness_list = []
        for result in results_list:
            fitness = fitness_fxn(result)
            fitness_list.append(fitness)
        
        return fitness_list

if __name__ == '__main__':
    af2()
