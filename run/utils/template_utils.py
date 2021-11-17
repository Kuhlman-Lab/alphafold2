"""Functions for getting templates and calculating template features."""

# Standard imports.
import dataclasses
import datetime
import glob
import os
import re
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from absl import logging

# AlphaFold imports.
from alphafold.common import residue_constants
from alphafold.data import mmcif_parsing
from alphafold.data import parsers
from alphafold.data.tools import kalign, utils
from alphafold.common import protein
from alphafold.data import templates


class TemplateHitFeaturizer:
  """A class for turning hhr hits to template features."""

  def __init__(
      self,
      mmcif_dir: str,
      max_template_date: str,
      max_hits: int,
      kalign_binary_path: str,
      release_dates_path: Optional[str],
      obsolete_pdbs_path: Optional[str],
      strict_error_check: bool = False):
    """Initializes the Template Search.

    Args:
      mmcif_dir: Path to a directory with mmCIF structures. Once a template ID
        is found by HHSearch, this directory is used to retrieve the template
        data.
      max_template_date: The maximum date permitted for template structures. No
        template with date higher than this date will be returned. In ISO8601
        date format, YYYY-MM-DD.
      max_hits: The maximum number of templates that will be returned.
      kalign_binary_path: The path to a kalign executable used for template
        realignment.
      release_dates_path: An optional path to a file with a mapping from PDB IDs
        to their release dates. Thanks to this we don't have to redundantly
        parse mmCIF files to get that information.
      obsolete_pdbs_path: An optional path to a file containing a mapping from
        obsolete PDB IDs to the PDB IDs of their replacements.
      strict_error_check: If True, then the following will be treated as errors:
        * If any template date is after the max_template_date.
        * If any template has identical PDB ID to the query.
        * If any template is a duplicate of the query.
        * Any feature computation errors.
    """
    self._mmcif_dir = mmcif_dir
    if not glob.glob(os.path.join(self._mmcif_dir, '*.cif')):
      logging.error('Could not find CIFs in %s', self._mmcif_dir)
      raise ValueError(f'Could not find CIFs in {self._mmcif_dir}')

    try:
      self._max_template_date = datetime.datetime.strptime(
          max_template_date, '%Y-%m-%d')
    except ValueError:
      raise ValueError(
          'max_template_date must be set and have format YYYY-MM-DD.')
    self._max_hits = max_hits
    self._kalign_binary_path = kalign_binary_path
    self._strict_error_check = strict_error_check

    if release_dates_path:
      logging.info('Using precomputed release dates %s.', release_dates_path)
      self._release_dates = templates._parse_release_dates(release_dates_path)
    else:
      self._release_dates = {}

    if obsolete_pdbs_path:
      logging.info('Using precomputed obsolete pdbs %s.', obsolete_pdbs_path)
      self._obsolete_pdbs = templates._parse_obsolete(obsolete_pdbs_path)
    else:
      self._obsolete_pdbs = {}

  def get_templates(
      self,
      query_sequence: str,
      query_release_date: Optional[datetime.datetime],
      hits: Sequence[parsers.TemplateHit]) -> templates.TemplateSearchResult:
    """Computes the templates for given query sequence (more details above)."""

    template_features = {}
    for template_feature_name in templates.TEMPLATE_FEATURES:
      template_features[template_feature_name] = []

    # Always use a max_template_date. Set to query_release_date minus 60 days
    # if that's earlier.
    template_cutoff_date = self._max_template_date
    if query_release_date:
      delta = datetime.timedelta(days=60)
      if query_release_date - delta < template_cutoff_date:
        template_cutoff_date = query_release_date - delta
      assert template_cutoff_date < query_release_date
    assert template_cutoff_date <= self._max_template_date

    num_hits = 0
    errors = []
    warnings = []

    for hit in sorted(hits, key=lambda x: x.sum_probs, reverse=True):
      # We got all the templates we wanted, stop processing hits.
      if num_hits >= self._max_hits:
        break

      result = templates._process_single_hit(
          query_sequence=query_sequence,
          hit=hit,
          mmcif_dir=self._mmcif_dir,
          max_template_date=template_cutoff_date,
          release_dates=self._release_dates,
          obsolete_pdbs=self._obsolete_pdbs,
          strict_error_check=self._strict_error_check,
          kalign_binary_path=self._kalign_binary_path)

      if result.error:
        errors.append(result.error)

      # There could be an error even if there are some results, e.g. thrown by
      # other unparseable chains in the same mmCIF file.
      if result.warning:
        warnings.append(result.warning)

      if result.features is None:
        logging.info('Skipped invalid hit %s, error: %s, warning: %s',
                     hit.name, result.error, result.warning)
      else:
        # Increment the hit counter, since we got features out of this hit.
        num_hits += 1
        for k in template_features:
          template_features[k].append(result.features[k])

    for name in template_features:
      if num_hits > 0:
        template_features[name] = np.stack(
            template_features[name], axis=0).astype(
              templates.TEMPLATE_FEATURES[name])
      else:
        # Make sure the feature has correct dtype even if empty.
        template_features[name] = np.array(
          [], dtype=templates.TEMPLATE_FEATURES[name])

    return templates.TemplateSearchResult(
        features=template_features, errors=errors, warnings=warnings)


def get_custom_template_features(template_path):

  def pdb_to_string(pdb_file):
    lines = []
    for line in open(pdb_file,"r"):
      if line[:6] == "HETATM" and line[17:20] == "MSE":
        line = "ATOM  "+line[6:17]+"MET"+line[20:]
      if line[:4] == "ATOM":
        lines.append(line)
    return "".join(lines)

  files = os.listdir(template_path)
  files = [file for file in files if file[-4:]=='.pdb']

  aatype = None
  atom_masks = None
  atom_positions = None
  domain_names = None
  for file in files:
    protein_obj = protein.from_pdb_string(
      pdb_to_string(os.path.join(template_path, file)), chain_id="A")
    if aatype == None:
      aatype = jax.nn.one_hot(protein_obj.aatype,22)[:][None]
    else:
      aatype = jnp.concatenate(
        (aatype, jax.nn.one_hot(protein_obj.aatype,22)[:][None]))

    if atom_masks == None:
      atom_masks = protein_obj.atom_mask[:][None]
    else:
      atom_masks = jnp.concatenate(
        (atom_masks, protein_obj.atom_mask[:][None]))
    
    if atom_positions == None:
      atom_positions = protein_obj.atom_positions[:][None]
    else:
      atom_positions = jnp.concatenate(
        (atom_positions, protein_obj.atom_positions[:][None]))
      
    if domain_names == None:
      domain_names = np.asarray(['None'])
    else:
      domain_names = jnp.concatenate(
        (domain_names, np.asarray(['None'])))
    
  template_features = {"template_aatype":aatype,
                         "template_all_atom_masks": atom_masks,
                         "template_all_atom_positions": atom_positions,
                         "template_domain_names": domain_names}
