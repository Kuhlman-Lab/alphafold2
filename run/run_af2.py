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
from features import getMonomerRawInputs, getMultimerRawInputs
from features import getChainFeatures, getInputFeatures
from utils.model_utils import getModelNames, getModelRunner, predictStructure
from utils.utils import compressed_pickle

from alphafold.common import protein
from alphafold.relax import relax

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

monomer_queries = qm.monomer_queries
multimer_queries = qm.multimer_queries
del qm

# Get raw model inputs.
raw_inputs_from_sequence, a3m_lines = getMonomerRawInputs(
    monomer_queries=monomer_queries,
    use_env=args.use_env,
    use_filter=args.use_filter,
    use_templates=args.use_templates,
    output_dir=args.output_dir)

raw_inputs_from_sequence = getMultimerRawInputs(
    multimer_queries=multimer_queries,
    use_env=args.use_env,
    use_filter=args.use_filter,
    use_templates=args.use_templates,
    output_dir=args.output_dir,
    raw_inputs=raw_inputs_from_sequence)

# Get random seeds.
seeds = getRandomSeeds(
    random_seed=args.random_seed,
    num_seeds=args.num_seeds)

# Get model names.
monomer_model_names = ()
if len(monomer_queries) > 0:
    monomer_model_names = getModelNames(
        mode='monomer', use_ptm=args.use_ptm, num_models=args.num_models)

multimer_model_names = ()
if len(multimer_queries) > 0:
    multimer_model_names = getModelNames(
        mode='multimer', num_models=args.num_models)

if args.use_amber:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0, tolerance=2.39, stiffness=10.0,
        exclude_residues=[], max_outer_iterations=3)
    
# Predict structures.
for model_name in monomer_model_names + multimer_model_names:
        model_runner = getModelRunner(
            model_name=model_name,
            num_ensemble=args.num_ensemble)

        for query in monomer_queries + multimer_queries:
            filename = query[0]
            if filename[-4:] == '.a3m':
                custom_a3m = a3m_lines[filename]
            else:
                custom_a3m = None
            
            sequences = query[-1]

            if isinstance(sequences, str):
                sequences = [sequences]

            features_for_chain = getChainFeatures(
                sequences=sequences,
                raw_inputs=raw_inputs_from_sequence,
                use_templates=args.use_templates
                custom_a3m_lines=custom_a3m)

            input_features = getInputFeatures(
                sequences=sequences,
                chain_features=features_for_chain,
                is_prokaryote=args.is_prokaryote)

            del filename, custom_a3m, sequences, features_for_chain

            for seed in seeds:
                if 'monomer' in model_name:
                    model_type = 'monomer'
                else:
                    model_type = 'multimer'

                result = predictStructure(
                    model_runner=model_runner,
                    feature_dict=input_features,
                    model_type=model_type,
                    random_seed=seed)

                if args.dont_write_pdbs:
                    unrelaxed_pdb = protein.to_pdb(result('unrelaxed_protein'])

                    unrelaxed_pred_path = os.path.join(
                        args.output_dir, 'unrelaxed_prediction.pdb')
                    with open(unrelaxed_pred_path, 'w') as f:
                        f.write(unrelaxed_pdb)

                        del unrelaxed_pdb, unrelaxed_pred_path
            
                if args.use_amber:
                    relaxed_pdb, _, _ = amber_relaxer.process(
                        prot=result['unrelaxed_protein'])

                    if not args.dont_write_pdbs:
                        relaxed_pred_path = os.path.join(
                            args.output_dir, 'relaxed_prediction.pdb')
                        with open(relaxed_pred_path, 'w') as f:
                            f.write(relaxed_pdb)

                        del relaxed_pred_path

                    del relaxed_pdb

                if args.compress_output:
                    results_path = os.path.join(
                        args.output_dir, 'results')
                    utils.compressed_pickle(results_path, results)

                    del results_path

                del result
