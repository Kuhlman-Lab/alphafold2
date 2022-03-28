""" Functions for setting up and running AF2 models. """

from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline


import random
import sys
import numpy as np
from typing import Tuple, Sequence, Optional, Dict

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
    
    if 'monomer' in model_name:
        cfg.data.common.num_recycle = num_recycle
        cfg.data.eval.num_ensemble = num_ensemble
    elif 'multimer' in model_name:
        cfg.model.num_ensemble_eval = num_ensemble

    cfg.model.num_recycle = num_recycle
    cfg.model.recycle_tol = recycle_tol
        
    params = data.get_model_haiku_params(model_name, params_dir)

    return model.RunModel(config=cfg, params=params, is_training=is_training)


def predictStructure(
        model_runner: model.RunModel,
        feature_dict: pipeline.FeatureDict,
        run_multimer: bool,
        random_seed: int = random.randrange(sys.maxsize)
        ) -> Dict[str, np.ndarray]:

    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)

    prediction, _ = model_runner.predict(
        processed_feature_dict, random_seed=random_seed)

    result = {}
    
    if 'predicted_aligned_error' in prediction:
        result['pae_output'] = (prediction['predicted_aligned_error'],
                                prediction['max_predicted_aligned_error'])

    result['ranking_confidence'] = prediction['ranking_confidence']
    result['plddt'] = prediction['plddt']
    result['structure_module'] = prediction['structure_module']

    b_factors = np.repeat(
        prediction['plddt'][:, None], residue_constants.atom_type_num, axis=-1)
    result['unrelaxed_protein'] = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction,
        b_factors=b_factors,
        remove_leading_feature_dimension=not run_multimer)
    
    return result
