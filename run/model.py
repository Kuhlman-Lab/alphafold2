""" Functions for setting up and running AF2 models. """

from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphafold.model.tf import shape_placeholders
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline

import tensorflow as tf
import random
import sys
import numpy as np
from typing import Tuple, Sequence, Optional, Dict, Mapping, Any

# (filename, sequence)
MonomerQuery = Tuple[str, str]

# (filename, oligomer_state, [sequences])
MultimerQuery = Tuple[str, str, Sequence[str]]


def getRandomSeeds(
        random_seed: Optional[int] = None,
        num_seeds: int = 1) -> Sequence[int]:

    seeds = []
    # If a random seed was provided, guarantee that it will be run.
    if random_seed != None:
        seeds.append(random_seed)

    # While we have less seeds than desired, get a new one.
    while len(seeds) < num_seeds:
        seeds.append(random.randrange(sys.maxsize))

    return seeds
        

def getModelNames(
        first_n_seqs: int, last_n_seqs: int,
        use_ptm: bool = True, num_models: int = 5,
        use_multimer = True, use_multimer_v1: bool = False, 
        use_multimer_v2: bool = False) -> Tuple[str]:

    include_monomer = False
    include_multimer = False
    if first_n_seqs == 1:
        include_monomer = True
    if last_n_seqs > 1:
        if use_multimer:
            include_multimer = True

    model_names = ()
    if include_monomer or not use_multimer:
        key = 'monomer_ptm' if use_ptm else 'monomer'
        monomer_models = config.MODEL_PRESETS[key]
        model_names += monomer_models[:num_models]
    if include_multimer:
        if use_multimer_v1:
            multimer_models = config.MODEL_PRESETS['multimer_v1']
        elif use_multimer_v2:
            multimer_models = config.MODEL_PRESETS['multimer_v2']
        else:
            multimer_models = config.MODEL_PRESETS['multimer_v3']
            
        model_names += multimer_models[:num_models]

    return model_names


def getModelRunner(
        model_name: str, num_ensemble: int = 1, is_training: bool = False,
        num_recycle: int = 3, recycle_tol: float = 0, stop_at_score: float = 100,
        rank_by: str = 'plddt', params_dir: str = '../alphafold/data') -> model.RunModel:

    cfg = config.model_config(model_name)
    
    if '_ptm' in model_name:
        cfg.data.common.num_recycle = num_recycle
        cfg.data.eval.num_ensemble = num_ensemble
    elif 'multimer' in model_name:
        cfg.model.num_ensemble_eval = num_ensemble

    cfg.model.num_recycle = num_recycle
    cfg.model.recycle_early_stop_tolerance = recycle_tol
        
    params = data.get_model_haiku_params(model_name, params_dir)

    return model.RunModel(config=cfg, params=params, is_training=is_training)


def getModelRunners(first_n_seqs: int, last_n_seqs: int, use_ptm: bool = True, num_models: int = 5,
                    use_multimer = True, use_templates: bool = True, use_v1: bool = False, num_ensemble: int = 1, 
                    is_training: bool = False, num_recycle: int = 3, recycle_tol: float = 0, stop_at_score: float = 100,
                    rank_by: str = 'plddt', params_dir: str = '../alphafold/data') -> Tuple[str]:

    include_monomer = False
    include_multimer = False
    if first_n_seqs == 1:
        include_monomer = True
    if last_n_seqs > 1:
        if use_multimer:
            include_multimer = True

    model_names = ()
    if include_monomer or not use_multimer:
        key = 'monomer_ptm' if use_ptm else 'monomer'
        monomer_models = config.MODEL_PRESETS[key]
        model_names += monomer_models[:num_models]
    if include_multimer:
        if use_v1:
            multimer_models = config.MODEL_PRESETS['multimer_v1']
        else:
            multimer_models = config.MODEL_PRESETS['multimer_v2']
            
        model_names += multimer_models[:num_models]

    if num_models >= 3:
        models_need_compilation = [1, 3] if use_templates else [3]
    else:
        models_need_compilation = [1]

    model_build_order = [3, 4, 5, 1, 2]
    model_runner_and_params_build_order = []
    model_runner = None

    ordered_model_names = []
    monomers = [name for name in model_names if 'multimer' not in name]
    monomer_nums = [int(name.split('_')[1]) for name in monomers]
    for model_number in model_build_order:
        if len(monomers) > 0:
            if model_number in monomer_nums:
                ordered_model_names.append(monomers[monomer_nums.index(model_number)])

    multimers = [name for name in model_names if 'multimer' in name]
    multimer_nums = [int(name.split('_')[1]) for name in multimers]
    for model_number in model_build_order:
        if len(multimers) > 0:
            if model_number in multimer_nums:
                ordered_model_names.append(multimers[multimer_nums.index(model_number)])

    for model_name in ordered_model_names:
        model_number = int(model_name.split('_')[1])
        if model_number in models_need_compilation:
            model_config = config.model_config(model_name)
            model_config.model.stop_at_score = float(stop_at_score)
            model_config.model.stop_at_score_ranker = rank_by
            if "_ptm" in model_name:
                model_config.data.common.num_recycle = num_recycle
                model_config.model.num_recycle = num_recycle
                model_config.data.eval.num_ensemble = num_ensemble
            elif "_multimer" in model_name:
                model_config.model.num_recycle = num_recycle
                if is_training:
                    model_config.model.num_ensemble_train = num_ensemble
                else:
                    model_config.model.num_ensemble_eval = num_ensemble
            model_runner = model.RunModel(
                model_config,
                data.get_model_haiku_params(
                    model_name=model_name,
                    data_dir=str(params_dir),
                ),
                is_training=is_training,
            )
        params = data.get_model_haiku_params(
            model_name=model_name, data_dir=str(params_dir)
        )
        # keep only parameters of compiled model
        params_subset = {}
        for k in model_runner.params.keys():
            params_subset[k] = params[k]

        model_runner.params = params_subset

        model_runner_and_params_build_order.append(
            (model_name, model_runner)
        )
    # reorder models
    monomer_model_runners = [model for model in model_runner_and_params_build_order if 'multimer' not in model[0]]
    multimer_model_runners = [model for model in model_runner_and_params_build_order if 'multimer' in model[0]]
    
    sorted_monomers = sorted(monomer_model_runners, key=lambda x: int(x[0].split('_')[1]))
    sorted_multimers = sorted(multimer_model_runners, key=lambda x: int(x[0].split('_')[1]))

    return sorted_monomers + sorted_multimers


NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

def make_fixed_size(protein: Mapping[str, Any], shape_schema, 
                    msa_cluster_size: int, extra_msa_size: int, num_res: int, 
                    num_templates: int = 0) -> model.features.FeatureDict:

    """Guesss at the MSA and sequence dimensions to make fixed size."""

    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    def _do_padding(k, v, parent_key=None):
        shape = list(v.shape)

        if parent_key:
            schema = shape_schema[parent_key][k]
        else:
            schema = shape_schema[k]

        assert len(shape) == len(schema), (
            f"Rank mismatch between shape and shape schema for {k}: "
            f"{shape} vs {schema}"
        )
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
        padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]
        if padding:
            crop_dims = []
            crop_needed = [pad[1] < 0 for pad in padding]
            if True in crop_needed:
                for i in range(len(padding)):
                    if crop_needed[i] == True:
                        padding[i] = (0, 0)
                        crop_dims.append(i)

            if crop_dims != []:
                for dim in crop_dims:
                    slice_size = [-1] * len(v.shape)
                    slice_size[dim] = pad_size_map[schema[dim]]
                    
                    v = tf.slice(v, [0] * len(v.shape), slice_size)

                    pad_size[dim] = pad_size_map[schema[dim]]

            if parent_key:
                protein[parent_key][k] = tf.pad(v, padding, name=f"pad_to_fixed_{k}")
                protein[parent_key][k].set_shape(pad_size)
            else:
                protein[k] = tf.pad(v, padding, name=f"pad_to_fixed_{k}")
                protein[k].set_shape(pad_size)
                
    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        elif k == 'prev':
            for sk, sv in protein[k].items():
                _do_padding(sk, sv, k)
        else:
            _do_padding(k, v)

    for k, v in protein.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                protein[k][sk] = np.asarray(sv)
        else:
            protein[k] = np.asarray(v)

    return protein


def batch_input(input_features: model.features.FeatureDict, model_runner: model.RunModel, model_name: str,
                crop_len: int, use_templates: bool, run_multimer: bool) -> model.features.FeatureDict:

    if not run_multimer:
        model_config = model_runner.config
        eval_cfg = model_config.data.eval
        crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

        max_msa_clusters = eval_cfg.max_msa_clusters
        max_extra_msa = model_config.data.common.max_extra_msa
        # template models
        if ("model_1" in model_name or "model_2" in model_name) and use_templates:
            pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
        else:
            pad_msa_clusters = max_msa_clusters

        max_msa_clusters = pad_msa_clusters
    else:
        MULTIMER_FEATS = {
            'aatype': [NUM_RES],
            'all_atom_mask': [NUM_RES, None],
            'all_atom_positions': [NUM_RES, None, None],
            'bert_mask': [NUM_MSA_SEQ, NUM_RES],
            'msa_mask': [NUM_MSA_SEQ, NUM_RES],
            'residue_index': [NUM_RES],
            'seq_length': [],
            'seq_mask': [NUM_RES],
            'template_aatype': [NUM_TEMPLATES, NUM_RES],
            'template_all_atom_mask': [NUM_TEMPLATES, NUM_RES, None],
            'template_all_atom_positions': [NUM_TEMPLATES, NUM_RES, None, None],
            'msa': [NUM_MSA_SEQ, NUM_RES],
            'num_alignments': [],
            'asym_id': [NUM_RES],
            'sym_id': [NUM_RES],
            'entity_id': [NUM_RES],
            'deletion_matrix': [NUM_MSA_SEQ, NUM_RES],
            'deletion_mean': [NUM_RES],
            'assembly_num_chains': [],
            'entity_mask': [NUM_RES],
            'num_templates': [],
            'cluster_bias_mask': [NUM_MSA_SEQ],
            'iter': [],
            'prev': {
                'prev_msa_first_row': [NUM_RES, None],
                'prev_pair': [NUM_RES, NUM_RES, None],
                'prev_pos': [NUM_RES, None, None]
            }
        }

        model_config = model_runner.config
        crop_feats = MULTIMER_FEATS

        max_msa_clusters = model_config.model.embeddings_and_evoformer.num_msa
        max_extra_msa = model_config.model.embeddings_and_evoformer.num_extra_msa

        # Multimer model separates msa and extra msa during runtime so need to combine
        max_msa_clusters += max_extra_msa

    # let's try pad (num_res + X)
    input_fix = make_fixed_size(input_features,
                                crop_feats,
                                msa_cluster_size=max_msa_clusters, # true_msa (4, 512, 68)
                                extra_msa_size=max_extra_msa, # extra_msa (4, 5120, 68)
                                num_res=crop_len, # aatype (4, 68)
                                num_templates=4,
    ) # template_mask (4, 4) second value
    return input_fix


def predictStructure(
        model_runner: model.RunModel,
        model_name: str,
        feature_dict: pipeline.FeatureDict,
        run_multimer: bool,
        use_templates: bool,
        random_seed: int = random.randrange(sys.maxsize),
        crop_size: Optional[int] = None,
        feature_dict_list = None,
        ) -> Dict[str, np.ndarray]:
    
    if feature_dict_list:
        #find max length of sequences and make that the crop size
        processed_feature_dict_list = {}
        max_seq_len = 0
        for fdict in feature_dict_list:
            processed_feature_dict = model_runner.process_features(fdict, random_seed=random_seed)
            if not run_multimer:
                seq_len = processed_feature_dict['aatype'].shape[1]
            else:
                seq_len = processed_feature_dict['seq_length']
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        crop_size = max_seq_len

        #loop over query features, make same size, batch them
        for fdict in feature_dict_list:
            processed_feature_dict = model_runner.process_features(fdict, random_seed=random_seed)
            processed_feature_dict = batch_input(processed_feature_dict, model_runner, model_name, crop_size, use_templates, run_multimer)

            for feature in processed_feature_dict:
                if feature not in processed_feature_dict_list:
                    processed_feature_dict_list[feature] = [processed_feature_dict[feature]]
                else:
                    np.append(processed_feature_dict_list[feature], processed_feature_dict[feature])

            for feature in processed_feature_dict_list:
                processed_feature_dict_list[feature] = np.array(processed_feature_dict_list[feature])

        predictions = model_runner.predict(processed_feature_dict_list, random_seed=random_seed)

        results = []
        for prediction in predictions:
            result = {}

            if 'predicted_aligned_error' in prediction:
                result['pae_output'] = (prediction['predicted_aligned_error'],
                                        prediction['max_predicted_aligned_error'])

            result['ranking_confidence'] = prediction['ranking_confidence']
            result['plddt'] = prediction['plddt']
            result['structure_module'] = prediction['structure_module']

            if 'ptm' in prediction:
                result['ptm'] = prediction['ptm']

            if 'iptm' in prediction:
                result['iptm'] = prediction['iptm']

            if crop_size:
                result['plddt'] = result['plddt'][:seq_len]
                result['pae_output'] = (result['pae_output'][0][:seq_len, :seq_len], result['pae_output'][1])

            b_factors = np.repeat(
                prediction['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
            result['unrelaxed_protein'] = protein.from_prediction(
                features=processed_feature_dict,
                result=prediction,
                b_factors=b_factors,
                remove_leading_feature_dimension=not run_multimer)

            results.append(result)

    else:
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)

        if not run_multimer:
            seq_len = processed_feature_dict['aatype'].shape[1]
        else:
            seq_len = processed_feature_dict['seq_length']

        if crop_size:
            processed_feature_dict = batch_input(processed_feature_dict, model_runner, model_name, crop_size, use_templates, run_multimer)

        prediction = model_runner.predict(
            processed_feature_dict, random_seed=random_seed)

        result = {}
        
        if 'predicted_aligned_error' in prediction:
            result['pae_output'] = (prediction['predicted_aligned_error'],
                                    prediction['max_predicted_aligned_error'])

        result['ranking_confidence'] = prediction['ranking_confidence']
        result['plddt'] = prediction['plddt']
        result['structure_module'] = prediction['structure_module']

        if 'ptm' in prediction:
            result['ptm'] = prediction['ptm']

        if 'iptm' in prediction:
            result['iptm'] = prediction['iptm']

        if crop_size:
            result['plddt'] = result['plddt'][:seq_len]
            result['pae_output'] = (result['pae_output'][0][:seq_len, :seq_len], result['pae_output'][1])

        b_factors = np.repeat(
            prediction['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
        result['unrelaxed_protein'] = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction,
            b_factors=b_factors,
            remove_leading_feature_dimension=not run_multimer)

        results = [result]
    
    return results
