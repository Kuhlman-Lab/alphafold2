""" Utility functions for setting up and running AF2 models. """

from alphafold.model import config
from alphafold.model import model
from alphafold.model import data

from alphafold.data import pipeline

import random
import sys
import numpy as np
from typing import Tuple, Sequence, Optional


def getRandomSeeds(
        random_seed: Optional[int],
        num_seeds: int) -> Sequence[int]:

    seeds = []
    # If a random seed was provided, guarantee that it will be run.
    if random_seed:
        seeds.append(random_seed)

    # While we have less seeds than desired, get a new one.
    while len(seeds) < num_seeds:
        seeds.append(random.randrange(sys.maxsize))

    return seeds
        

def getModelNames(
        mode: str, use_ptm: bool = True, num_models: int = 5) -> Tuple[str]:

    if mode == 'monomer':
        key = 'monomer_ptm' if use_ptm else 'monomer'
    elif mode == 'multimer':
        key = 'multimer'

    model_names = config.MODEL_PRESETS[key]
    model_names = model_names[:num_models]

    return model_names


def getModelRunner(
        model_name: str, num_ensemble: int = 1, is_training: bool = False,
        num_recycle: int = 3, recycle_tol: float = 0,
        params_dir: str = '../alphafold/data') -> model.RunModel:

    cfg = config.model_config(model_name)

    if 'monomer' in model_name:
        cfg.data.eval.num_ensemble = num_ensemble
    elif 'multimer' in model_name:
        cfg.model.num_ensemble_eval = num_ensemble

    cfg.model.num_recycle = num_recycle
    cfg.model.recycle_tol = recycle_tol
        
    params = data.get_model_haiku_params(model_name, params_dir)

    return model.RunModel(cfg, params, is_training)


def predictStructure(
        model_runner: model.RunModel,
        feature_dict: pipeline.FeatureDict,
        model_type: str
        random_seed: int = random.randrange(sys.maxsize)
        ) -> Dict[str, np.ndarray]:

    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)

    prediction = model_runner.predict(
        processed_feature_dict, random_seed=random_seed)

    result = {}
    
    if 'predicted_aligned_error' in prediction:
        result['pae_output'] = (prediction['predicted_aligned_error'],
                                prediction['max_predicted_aligned_error'])

    result['ranking_confidence'] = prediction['ranking_confidence']
    result['plddt'] = prediction['plddt']
    result['structure_module'] = prediction['structure_model']

    final_atom_mask = prediction['structure_module']['final_atom_mask']
    b_factors = prediction['plddts'][:, None] * final_atom_mask
    result['unrelaxed_protein'] = protein.from_prediction(
        processed_feature_dict,
        prediction,
        b_factors=b_factors,
        remove_leading_feature_dimension=(
            model_type == 'monomer'))
    
    return result
