"""Functions for getting templates and calculating template features."""

# Standard imports.
import dataclasses
import datetime
import glob
import os
import re
import subprocess
import shutil
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from absl import logging
from io import StringIO
from pathlib import Path
from Bio.PDB import MMCIF2Dict, PDBParser, MMCIFIO, MMCIFParser
from Bio.PDB.Polypeptide import standard_aa_names

# AlphaFold imports.
from alphafold.common import residue_constants
from alphafold.data import mmcif_parsing
from alphafold.data import parsers
from alphafold.data.tools import kalign, utils
from alphafold.common import protein
from alphafold.data import templates

logger = logging.get_absl_logger()

CIF_REVISION_DATE = """loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 1971-01-01
#\n"""

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


### begin section copied from Bio.PDB
mmcif_order = {
    "_atom_site": [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_alt_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "pdbx_PDB_ins_code",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_formal_charge",
        "auth_seq_id",
        "auth_comp_id",
        "auth_asym_id",
        "auth_atom_id",
        "pdbx_PDB_model_num",
    ]
}


class CFMMCIFIO(MMCIFIO):
    def _save_dict(self, out_file):
        # Form dictionary where key is first part of mmCIF key and value is list
        # of corresponding second parts
        key_lists = {}
        for key in self.dic:
            if key == "data_":
                data_val = self.dic[key]
            else:
                s = re.split(r"\.", key)
                if len(s) == 2:
                    if s[0] in key_lists:
                        key_lists[s[0]].append(s[1])
                    else:
                        key_lists[s[0]] = [s[1]]
                else:
                    raise ValueError("Invalid key in mmCIF dictionary: " + key)

        # Re-order lists if an order has been specified
        # Not all elements from the specified order are necessarily present
        for key, key_list in key_lists.items():
            if key in mmcif_order:
                inds = []
                for i in key_list:
                    try:
                        inds.append(mmcif_order[key].index(i))
                    # Unrecognised key - add at end
                    except ValueError:
                        inds.append(len(mmcif_order[key]))
                key_lists[key] = [k for _, k in sorted(zip(inds, key_list))]

        # Write out top data_ line
        if data_val:
            out_file.write("data_" + data_val + "\n#\n")
            ### end section copied from Bio.PDB
            # Add poly_seq as default MMCIFIO doesn't handle this
            out_file.write(
                """loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
#\n"""
            )
            poly_seq = []
            chain_idx = 1
            for model in self.structure:
                for chain in model:
                    res_idx = 1
                    for residue in chain:
                        poly_seq.append(
                            (chain_idx, res_idx, residue.get_resname(), "n")
                        )
                        res_idx += 1
                    chain_idx += 1
            for seq in poly_seq:
                out_file.write(f"{seq[0]} {seq[1]} {seq[2]}  {seq[3]}\n")
            out_file.write("#\n")
            out_file.write(
                """loop_
_chem_comp.id
_chem_comp.type
#\n"""
            )
            for three in standard_aa_names:
                out_file.write(f'{three} "peptide linking"\n')
            out_file.write("#\n")
            out_file.write(
                """loop_
_struct_asym.id
_struct_asym.entity_id
#\n"""
            )
            chain_idx = 1
            for model in self.structure:
                for chain in model:
                    out_file.write(f"{chain.get_id()} {chain_idx}\n")
                    chain_idx += 1
            out_file.write("#\n")

        ### begin section copied from Bio.PDB
        for key, key_list in key_lists.items():
            # Pick a sample mmCIF value, which can be a list or a single value
            sample_val = self.dic[key + "." + key_list[0]]
            n_vals = len(sample_val)
            # Check the mmCIF dictionary has consistent list sizes
            for i in key_list:
                val = self.dic[key + "." + i]
                if (
                    isinstance(sample_val, list)
                    and (isinstance(val, str) or len(val) != n_vals)
                ) or (isinstance(sample_val, str) and isinstance(val, list)):
                    raise ValueError(
                        "Inconsistent list sizes in mmCIF dictionary: " + key + "." + i
                    )
            # If the value is a single value, write as key-value pairs
            if isinstance(sample_val, str) or (
                isinstance(sample_val, list) and len(sample_val) == 1
            ):
                m = 0
                # Find the maximum key length
                for i in key_list:
                    if len(i) > m:
                        m = len(i)
                for i in key_list:
                    # If the value is a single item list, just take the value
                    if isinstance(sample_val, str):
                        value_no_list = self.dic[key + "." + i]
                    else:
                        value_no_list = self.dic[key + "." + i][0]
                    out_file.write(
                        "{k: <{width}}".format(k=key + "." + i, width=len(key) + m + 4)
                        + self._format_mmcif_col(value_no_list, len(value_no_list))
                        + "\n"
                    )
            # If the value is more than one value, write as keys then a value table
            elif isinstance(sample_val, list):
                out_file.write("loop_\n")
                col_widths = {}
                # Write keys and find max widths for each set of values
                for i in key_list:
                    out_file.write(key + "." + i + "\n")
                    col_widths[i] = 0
                    for val in self.dic[key + "." + i]:
                        len_val = len(val)
                        # If the value requires quoting it will add 2 characters
                        if self._requires_quote(val) and not self._requires_newline(
                            val
                        ):
                            len_val += 2
                        if len_val > col_widths[i]:
                            col_widths[i] = len_val
                # Technically the max of the sum of the column widths is 2048

                # Write the values as rows
                for i in range(n_vals):
                    for col in key_list:
                        out_file.write(
                            self._format_mmcif_col(
                                self.dic[key + "." + col][i], col_widths[col] + 1
                            )
                        )
                    out_file.write("\n")
            else:
                raise ValueError(
                    "Invalid type in mmCIF dictionary: " + str(type(sample_val))
                )
            out_file.write("#\n")
            ### end section copied from Bio.PDB
            out_file.write(CIF_REVISION_DATE)


def validate_and_fix_mmcif(cif_file: Path):
  """ Validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
  # check that required poly_seq and revision_date fields are present
  cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
  required = [
    "_chem_comp.id",
    "_chem_comp.type",
    "_struct_asym.id",
    "_struct_asym.entity_id",
    "_entity_poly_seq.mon_id",
  ]
  for r in required:
    if r not in cif_dict:
      cif_dict[r] = '?'
      #raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
  if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
    logger.info(
      f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
    )
    shutil.copy2(cif_file, str(cif_file) + ".bak")
    with open(cif_file, "a") as f:
      f.write(CIF_REVISION_DATE)


def convert_pdb_to_mmcif(pdb_file: Path):
  """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
  id = pdb_file.stem
  cif_file = pdb_file.parent.joinpath(f"{id}.cif")
  if cif_file.is_file():
    return
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure(id, pdb_file)
  cif_io = CFMMCIFIO()
  cif_io.set_structure(structure)
  cif_io.save(str(cif_file))


def mk_hhsearch_db(template_dir: str, proc_id: Optional[int] = None):
  template_path = Path(template_dir)

  cif_files = template_path.glob("*.cif")
  for cif_file in cif_files:
    validate_and_fix_mmcif(cif_file)

  pdb_files = template_path.glob("*.pdb")
  for pdb_file in pdb_files:
    convert_pdb_to_mmcif(pdb_file)

  pdb70_db_files = template_path.glob("pdb70*")
  for f in pdb70_db_files:
    os.remove(f)

  prefix = ''
  if proc_id:
    prefix = f'{proc_id}_'
    
  with open(template_path.joinpath(f"{prefix}pdb70_a3m.ffdata"), "w") as a3m, open(
    template_path.joinpath(f"{prefix}pdb70_cs219.ffindex"), "w"
  ) as cs219_index, open(
    template_path.joinpath(f"{prefix}pdb70_a3m.ffindex"), "w"
  ) as a3m_index, open(
    template_path.joinpath(f"{prefix}pdb70_cs219.ffdata"), "w"
  ) as cs219:

    id = 1000000
    index_offset = 0
    cif_files = template_path.glob("*.cif")
    for cif_file in cif_files:
      with open(cif_file) as f:
        cif_string = f.read()
      cif_fh = StringIO(cif_string)
      parser = MMCIFParser(QUIET=True)
      structure = parser.get_structure("temp", cif_fh)
      models = list(structure.get_models())
      if len(models) != 1:
        raise ValueError(
          f"Only single model PDBs are supported. Found {len(models)} models."
        )
      model = models[0]
      for chain in model:
        amino_acid_res = []
        for res in chain:
          if res.id[2] != " ":
            raise ValueError(
              f"PDB ({cif_file}) contains an insertion code at chain {chain.id} and residue "
              f"index {res.id[1]}. These are not supported."
            )
          amino_acid_res.append(
            residue_constants.restype_3to1.get(res.resname, "X")
          )

        protein_str = "".join(amino_acid_res)
        a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
        a3m_str_len = len(a3m_str)
        a3m_index.write(f"{id}\t{index_offset}\t{a3m_str_len}\n")
        cs219_index.write(f"{id}\t{index_offset}\t{len(protein_str)}\n")
        index_offset += a3m_str_len
        a3m.write(a3m_str)
        cs219.write("\n\0")
        id += 1
