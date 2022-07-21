""" Full AlphaFold protein structure prediction script."""

# Standard imports.
import os
import sys
import logging
import time
from typing import Sequence, Union, Optional
from functools import partial

# Update PATH.
sys.path.append('~/.miniconda3/envs/af2/lib/python3.7/site-packages')
sys.path.append('../content/alphafold')

# Custom imports.
from setup import getAF2Parser, QueryManager, getOutputDir, determine_weight_directory
from utils.utils import compressed_pickle, full_pickle

# Global constants.
MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def af2_init(proc_id: int, arg_file: str, lengths: Sequence[Union[int, Sequence[int]]], fitness_fxn):
    print('initialization of process', proc_id)
    
    os.environ['TF_FORCE_UNITED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(proc_id)

    import jax
    from features import (
        getRawInputs, getChainFeatures, getInputFeatures)
    from setup import (getAF2Parser, QueryManager, getOutputDir)
    from model import (getModelNames, getModelRunner, predictStructure)
    from utils.query_utils import generate_random_sequences

    parser = getAF2Parser()
    args = parser.parse_args([f'@{arg_file}'])
    if not args.params_dir:
        args.params_dir = determine_weight_directory()

    output_dir = getOutputDir(out_dir=args.output_dir)

    # Generate mock sequences
    sequences = generate_random_sequences(lengths, 1, aalist=['A'])[0]
    #print('run_af2::af2_init:', sequences)

    qm = QueryManager(
        sequences=sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        max_multimer_length=args.max_multimer_length)
    qm.parse_files()
    qm.parse_sequences()
    queries = qm.queries
    #print('run_af2::af2_init:', queries)
    del qm
    
    raw_inputs = getRawInputs(
        queries=queries,
        msa_mode='single_sequence',
        use_templates=False,
        output_dir=output_dir,
        design_run=args.design_run,
        proc_id=proc_id)
    #print('run_af2::af2_init:', raw_inputs)

    model_names = getModelNames(
        first_n_seqs=len(queries[0][1]),
        last_n_seqs=len(queries[-1][1]),
        use_ptm=args.use_ptm, num_models=args.num_models,
        use_multimer=not args.no_multimer_models,
        use_v1=args.use_multimer_v1)
    
    query_features = []
    for query_idx, query in enumerate(queries):
        sequences = query[1]
    
        features_for_chain = getChainFeatures(
            sequences=sequences,
            raw_inputs=raw_inputs,
            use_templates=args.use_templates,
            use_multimer=not args.no_multimer_models)

        input_features = getInputFeatures(
            sequences=sequences,
            chain_features=features_for_chain,
            use_multimer=not args.no_multimer_models)

        query_features.append( (sequences, input_features) )

    results_list = []
    for model_name in model_names:
        model_runner = getModelRunner(
            model_name=model_name,
            num_ensemble=args.num_ensemble,
            is_training=args.is_training,
            num_recycle=args.max_recycle,
            recycle_tol=args.recycle_tol,
            params_dir=args.params_dir)

        run_multimer = False
        if 'multimer' in model_name:
            run_multimer = True
        
        for query in query_features:
            sequences = query[0]

            if len(sequences) > 1 and not run_multimer:
                if not args.no_multimer_models:
                    continue
            elif len(sequences) == 1 and run_multimer:
                continue

            input_features = query[1]

            del sequences

            t = time.time()
            result = predictStructure(
                model_runner=model_runner,
                feature_dict=input_features,
                run_multimer=run_multimer)
            print(f'Model {model_name} took {time.time()-t} sec on GPU {proc_id}.')

    af2_partial = partial(af2, arg_file=arg_file, proc_id=proc_id, fitness_fxn=fitness_fxn, compiled_runners=[(model_names[-1], model_runner)])

    return af2_partial
    

def af2(sequences: Optional[Sequence[Sequence[str]]] = [],
        arg_file: Optional[str] = None,
        proc_id: Optional[int] = None,
        fitness_fxn = None,
        compiled_runners = None) -> Optional[Sequence[float]]:

    parser = getAF2Parser()
    if arg_file != None:
        args = parser.parse_args([f'@{arg_file}'])
    else:
        args = parser.parse_args(sys.argv[1:])
    del parser
    if not args.params_dir:
        args.params_dir = determine_weight_directory()
    
    output_dir = getOutputDir(out_dir=args.output_dir)
    
    # Set up logger
    if not args.no_logging and not args.design_run:
        logging.basicConfig(filename=os.path.join(output_dir, 'prediction.log'),
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join(output_dir, 'prediction.log'))
    logger = logging.getLogger('run_af2')

    # Update environmental variables.
    if proc_id != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(proc_id)
    
    # Get all AF2 imports.
    import jax
    from alphafold.common import protein
    from alphafold.relax import relax
    from features import (
        getRawInputs, getChainFeatures, getInputFeatures)
    from model import (
        getRandomSeeds, predictStructure, getModelRunners)

    # Log devices
    devices = jax.local_devices()
    NUM_DEVICES = len(devices)
    logger.info(f'Running JAX with {NUM_DEVICES} devices.')
    
    # Set up timing dictionary.
    timings = {}
    t_all = time.time()
     
    # Parse queries.
    if compiled_runners is None:
        input_dir = args.input_dir
    else:
        input_dir = ''

    qm = QueryManager(
        input_dir=input_dir,
        sequences=sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        max_multimer_length=args.max_multimer_length)
    qm.parse_files()
    qm.parse_sequences()

    queries = qm.queries
    logger.info(f'Queries have been parsed. {len(queries)} queries found.')
    #print(queries)
    del qm

    # Get raw model inputs.
    t_0 = time.time()
    raw_inputs_from_sequence = getRawInputs(
        queries=queries,
        msa_mode=args.msa_mode,
        use_templates=args.use_templates,
        custom_msa_path=args.custom_msa_path,
        insert_msa_gaps=args.insert_msa_gaps,
        custom_template_path=args.custom_template_path,
        output_dir=output_dir,
        design_run=args.design_run,
        proc_id=proc_id)
    #print(raw_inputs_from_sequence)
        
    timings['raw_inputs'] = time.time() - t_0
    logger.info(f'Raw inputs have been generated. Took '
                f'{timings["raw_inputs"]:.2f} seconds.')

    # Get random seeds.
    seeds = getRandomSeeds(
        random_seed=args.random_seed,
        num_seeds=args.num_seeds)

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
            use_multimer=not args.no_multimer_models)

        input_features = getInputFeatures(
            sequences=sequences,
            chain_features=features_for_chain,
            use_multimer=not args.no_multimer_models)
        
        timings[f'features_{query_idx}'] = time.time() - t_0
        logger.info(f'Features for query {query_idx} have been generated. Took '
                    f'{timings[f"features_{query_idx}"]} seconds.')

        query_features.append( (prefix, sequences, input_features) )
        
    #print(query_features)

    # Get model names.
    if not compiled_runners:
        model_names_and_runners = getModelRunners(
            first_n_seqs=len(queries[0][1]),
            last_n_seqs=len(queries[-1][1]),
            use_ptm=args.use_ptm, num_models=args.num_models,
            use_multimer=not args.no_multimer_models,
            use_templates=args.use_templates,
            use_v1=args.use_multimer_v1,
            num_ensemble=args.num_ensemble,
            is_training=args.is_training,
            num_recycle=args.max_recycle,
            recycle_tol=args.recycle_tol,
            params_dir=args.params_dir)
    else:
        model_names_and_runners = compiled_runners

    results_list = []
    # Predict structures.
    for model_name, model_runner in model_names_and_runners:
        logger.info(f'Obtained model runner for {model_name}.')

        if 'multimer' in model_name:
            run_multimer = True
        else:
            run_multimer = False

        for query_idx, query in enumerate(query_features):
            prefix = query[0] + f'_{model_name}'
            sequences = query[1]

            if len(sequences) > 1 and not run_multimer:
                if not args.no_multimer_models:
                    continue
            elif len(sequences) == 1 and run_multimer:
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
                if proc_id is not None:
                    print(f'Model {model_name} took {timings[f"predict_{model_name}_{seed_idx}"]} sec on GPU {os.environ["CUDA_VISIBLE_DEVICES"]}.')
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
                        
                del jobname, result
            
            # end for seed in seeds
            del prefix, input_features

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
    os.environ['TF_FORCE_UNITED_MEMORY'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
    
    af2()
