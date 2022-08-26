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
        use_multimer = True, use_v1: bool = False) -> Tuple[str]:

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

    return model_names


def getModelRunner(
        model_name: str, num_ensemble: int = 1, is_training: bool = False,
        num_recycle: int = 3, recycle_tol: float = 0, stop_at_score: float = 100,
        rank_by: str = 'plddt', params_dir: str = '../alphafold/data') -> model.RunModel:

    cfg = config.model_config(model_name)
    cfg.model.stop_at_score = float(stop_at_score)
    cfg.model.stop_at_score_ranker = rank_by
    
    if '_ptm' in model_name:
        cfg.data.common.num_recycle = num_recycle
        cfg.data.eval.num_ensemble = num_ensemble
    elif 'multimer' in model_name:
        cfg.model.num_ensemble_eval = num_ensemble

    cfg.model.num_recycle = num_recycle
    cfg.model.recycle_tol = recycle_tol
        
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

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)

        schema = shape_schema[k]

        assert len(shape) == len(schema), (
            f"Rank mismatch between shape and shape schema for {k}: "
            f"{shape} vs {schema}"
        )
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
        padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]

        if padding:
            protein[k] = tf.pad(v, padding, name=f"pad_to_fixed_{k}")
            protein[k].set_shape(pad_size)

    return {k: np.asarray(v) for k, v in protein.items()}


def batch_input(input_features: model.features.FeatureDict, model_runner: model.RunModel, model_name: str,
                crop_len: int, use_templates: bool) -> model.features.FeatureDict:

    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa
    # template models
    if (model_name == "model_1" or model_name == "model_2") and use_templates:
        pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
    else:
        pad_msa_clusters = max_msa_clusters

    max_msa_clusters = pad_msa_clusters

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
        crop_size: Optional[Sequence[Optional[int]]] = None,
        ) -> Dict[str, np.ndarray]:
    
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)

    if not run_multimer:
        seq_len = processed_feature_dict['aatype'].shape[1]

    if crop_size and not run_multimer:
        processed_feature_dict = batch_input(processed_feature_dict, model_runner, model_name, crop_size, use_templates)

    prediction, _ = model_runner.predict(
        processed_feature_dict, random_seed=random_seed)

    result = {}
    
    if 'predicted_aligned_error' in prediction:
        result['pae_output'] = (prediction['predicted_aligned_error'],
                                prediction['max_predicted_aligned_error'])

    result['ranking_confidence'] = prediction['ranking_confidence']
    result['plddt'] = prediction['plddt']
    result['structure_module'] = prediction['structure_module']

    if crop_size and not run_multimer:
        result['plddt'] = result['plddt'][:seq_len]
        result['pae_output'] = (result['pae_output'][0][:seq_len, :seq_len], result['pae_output'][1])

    b_factors = np.repeat(
        prediction['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
    result['unrelaxed_protein'] = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction,
        b_factors=b_factors,
        remove_leading_feature_dimension=not run_multimer)
    
    return result
