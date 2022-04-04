import os
import sys
import re
import lzma
import copy
import argparse
import subprocess
import textwrap
from itertools import combinations
from typing import Iterable, Dict, Tuple, Optional, Union, List

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm, trange

# Required dependencies
try:
    import psutil
except ImportError as e:
    psutil = e
try:
    from lxml import etree
except ImportError as e:
    etree = e
try:
    import Bio
    from Bio import AlignIO, Phylo, SeqRecord
    from Bio.Align import substitution_matrices
    from Bio.Align.substitution_matrices import Array as BioArray
    from Bio.Phylo import BaseTree
    from Bio.Phylo.TreeConstruction import TreeConstructor
except ImportError as e:
    Bio = e

from papyrus_scripts.preprocess import *
from papyrus_scripts.reader import *
from papyrus_scripts.utils.IO import process_data_version


# Handle missing librairies
missing = []
for name, dependency in [('psutil', psutil),
                         ('lxml', etree),
                         ('biopython', Bio)]:
    if isinstance(dependency, ImportError):
        missing.append(name)
if len(missing):
    raise ImportError(f'Some required dependencies are missing:\n\t{", ".join(missing)}')


# Handle pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'


def dataframe_to_fasta(df: pd.DataFrame, out_file: str) -> None:
    """Write Papyrus protein data from a dataframe to a FASTA file.

    :param df: dataframe of Papyrus targets
    :param out_file: path to FASTA file to be written
    """
    with open(out_file, 'w') as oh:
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            oh.write('>' + row['target_id'].replace(' ', '_') + '\n')
            oh.write(textwrap.fill(row['Sequence'].replace('U', 'C'), width=80) + '\n')


def run_kalign(input_fasta: str, output_fasta: str, kalign_path: str) -> None:
    """Run Kalign to obtain a multiple sequence alignment.

    Prefer using Kalign 3 for fast and accurate results:
    Timo Lassmann, Kalign 3: multiple sequence alignment of large datasets,
    Bioinformatics, Volume 36, Issue 6, 15 March 2020, Pages 1928–1929,
    https://doi.org/10.1093/bioinformatics/btz795

    :param input_fasta: path to FASTA file contining sequence to align
    :param output_fasta: FASTA file of aligned sequences to be written out
    :param kalign_path: path to Kalign 3
    """
    process = subprocess.run([kalign_path, '-i', input_fasta, '-o', output_fasta], capture_output=True)
    if len(process.stderr) != 0:
        raise ChildProcessError(f'Kalign returned with the following error:\n{process.stderr.decode()}')
    print(process.stdout.decode())


def get_distance_from_aligned_seqs(seq1: SeqRecord.SeqRecord,
                                   seq2: SeqRecord.SeqRecord,
                                   scoring_matrix: Optional[Union[str, BioArray]],
                                   skip_letters: Optional[Iterable[str]] = None):
    """Get the distance (0: equal, to 1: completely different) between two aligned sequences.

    :param seq1: Biopython SeqRecord of aligned sequence
    :param seq2: Biopython SeqRecord of aligned sequence
    :param scoring_matrix: scoring matrix or 'identity'
    """
    available_matrices = substitution_matrices.load()
    if isinstance(scoring_matrix, str):
        scoring_matrix = scoring_matrix.upper()
        if scoring_matrix != 'IDENTITY' and scoring_matrix not in available_matrices:
            raise ValueError(f'{scoring_matrix} scoring matrix not available')
    elif not isinstance(scoring_matrix, BioArray):
        raise ValueError('type of scoring matrix not supported')
    if isinstance(scoring_matrix, str) and scoring_matrix == 'IDENTITY':
        if skip_letters is None:
            skip_letters = ()
        score = sum(
            l1 == l2
            for l1, l2 in zip(seq1, seq2)
            if l1 not in skip_letters and l2 not in skip_letters
        )
        max_score = len(seq1)
    else:
        if isinstance(scoring_matrix, str):
            scm = substitution_matrices.load(scoring_matrix)
        else:
            scm = scoring_matrix
        if skip_letters is None:
            skip_letters = ('-', '*')
        max_score1 = 0
        max_score2 = 0
        score = 0
        max_score = 0
        for l1, l2 in zip(seq1, seq2):
            if l1 in skip_letters or l2 in skip_letters:
                continue
            max_score1 += scm[l1, l1]
            max_score2 += scm[l2, l2]
            score += scm[l1, l2]
        max_score = max(max_score1, max_score2)
    return (seq1.id, seq2.id, 1) if max_score == 0 else (seq1.id, seq2.id, 1 - (score * 1.0 / max_score))


class _Matrix:
    """Base class for distance matrix or scoring matrix.

    Reimplementation of Biopython's Bio.Phylo.TreeContruction._Matrix
    to use NumPy's dense matrices
    """

    def __init__(self, names: List[str],
                 matrix: np.ndarray = None):
        """Initialize matrix.

        :param names: a list of names
        :param matrix: a NumPy dense matrix
        """
        # check names
        if isinstance(names, Iterable) and all(isinstance(s, str) for s in names):
            if any('$' in s for s in names):
                raise ValueError("Names must not contain any '$' sign")
            elif len(set(names)) == len(names):
                self.names = names[:]
            else:
                raise ValueError("Duplicate names found")
        else:
            raise TypeError("'names' should be a list of strings")

        if not isinstance(matrix, np.ndarray):
            raise TypeError("'matrix' should be a numpy matrix")
        self.matrix = matrix

    def __getitem__(self, item):
        """Access value(s) by index(ices) or name(s).

        :param item: dm[i]                -> get a value list from the given 'i' to others;
                     dm[i, j]             -> get the value between 'i' and 'j';
                     dm['name']           -> map name to index first
                     dm['name1', 'name2'] -> map name to index first
        """
        # Handle single indexing
        if isinstance(item, str):
            if item in self.names:
                index = self.names.index(item)
                return self.matrix[index, :]
            else:
                raise ValueError("Item not found.")
        # Handle double indexing
        elif len(item) == 2:
            if all(isinstance(i, str) for i in item):
                row_name, col_name = item
                if row_name in self.names and col_name in self.names:
                    row_index = self.names.index(row_name)
                    col_index = self.names.index(col_name)
                    return self.matrix[row_index, col_index]
                else:
                    raise ValueError("Item not found.")
            # let NumPy handle any other type
            row_index, col_index = item
            return self.matrix[row_index, col_index]
        else:
            raise ValueError("'item' must either be a str or two str|int|slice values.")

    def __setitem__(self, item, value):
        """Set value by the index(ices) or name(s).

        :param item: dm[1] = [1, 0, 3] -> set values for row 1 then for column 1;
                     dm[i, j] = 2      -> set the value at row i and column j
                     dm[i, k:] = 2     -> set the value at row i and columns up to k
        :param value: value to be set
        """
        if isinstance(item, str):
            if item in self.names:
                index = self.names.index(item)
                self.matrix[index, :] = self.matrix[:, index] = value
            else:
                raise ValueError("Item not found.")
        # Handle double indexing
        elif len(item) == 2:
            if all(isinstance(i, str) for i in item):
                row_name, col_name = item
                if row_name in self.names and col_name in self.names:
                    row_index = self.names.index(row_name)
                    col_index = self.names.index(col_name)
                    self.matrix[row_index, col_index] = value
                else:
                    raise ValueError("Item not found.")
            # let NumPy handle any other type
            row_index, col_index = item
            self.matrix[row_index, col_index] = value
        else:
            raise ValueError("'item' must either be a str or two str|int|slice values.")

    def __delitem__(self, item):
        """Delete related distances by the index or name.

        :param item: index or name of the row/col to be removed
        """
        index = None
        if isinstance(item, int):
            index = item
        elif isinstance(item, str):
            if item in self.names:
                index = self.names.index(item)
            else:
                raise ValueError("Item not found.")
        else:
            print(item, type(item))
            raise TypeError("Invalid index type.")
        # remove distances related to index
        self.matrix = np.delete(self.matrix, index, axis=0)
        self.matrix = np.delete(self.matrix, index, axis=1)
        # remove name
        del self.names[index]

    def insert(self, item, value):
        """Insert distances given the name and value.

        :param item: index or name  of a row/col to be inserted
        :param value: a row/col of values to be inserted
        """
        if isinstance(item, int):
            index = item
        elif isinstance(item, str):
            if item in self.names:
                index = self.names.index(item)
            else:
                raise ValueError("Item not found.")
        else:
            raise ValueError("'item' must either be a str or two str|int|slice values.")
        self.matrix = np.insert(self.matrix, index, value[:index] + value[index + 1:], 0)
        self.matrix = np.insert(self.matrix, index, value, 1)

    def __len__(self):
        """Get Matrix length."""
        return len(self.names)

    def __repr__(self):
        """Return Matrix as a string."""
        return (f'{self.__class__.__name__}(names={len(self.names)})')

    def __eq__(self, other):
        """Check equality of Matrix with another."""
        if not isinstance(other, _Matrix):
            raise TypeError("'other' must be a Matrix object")
        if self.names != other.names:
            return False
        return np.all(np.equal(self.matrix, other.matrix))


class DistanceMatrix(_Matrix):
    """Distance matrix class that can be used for distance based tree algorithms.
    All diagonal elements will be zero no matter what the users provide.

    Reimplementation of Biopython's Bio.Phylo.TreeContruction._Matrix
    to use NumPy's dense matrices
    """

    def __init__(self, names: List[str], matrix: np.ndarray = None):
        """Initialize the class.

        :param names: a list of names
        :param matrix: a NumPy dense matrix
        """
        _Matrix.__init__(self, names, matrix)
        self._set_zero_diagonal()

    def __setitem__(self, item, value):
        """Set DistanceMatrix's items to values.

         :param item: dm[1] = [1, 0, 3] -> set values for row 1 then for column 1;
                     dm[i, j] = 2      -> set the value at row i and column j
                     dm[i, k:] = 2     -> set the value at row i and columns up to k
        :param value: value to be set
        """
        _Matrix.__setitem__(self, item, value)
        self._set_zero_diagonal()

    def _set_zero_diagonal(self):
        """Set all diagonal elements to zero."""
        np.fill_diagonal(self.matrix, 0)


class DistanceTreeConstructor(TreeConstructor):
    """Distance based tree constructor.

    Reimplementation of Biopython's Bio.Phylo.TreeContruction._Matrix
    to use NumPy's dense matrices
    """

    methods = ["nj", "upgma"]

    def __init__(self, distance_matrix: DistanceMatrix, method="nj"):
        """Initialize the class.

        :param distance_matrix: DistanceMatrix to build the phylogenetic tree from
        :param method: clustering method used to build tree:
                         - 'nj'   : Neighbour-Joining (default)
                         - 'upgma': Unweighted Pair Group Method with Arithmetic mean
        """
        if not isinstance(distance_matrix, DistanceMatrix):
            raise TypeError("distance_matrix must be a DistanceMatrix object.")
        self.dm = distance_matrix
        if isinstance(method, str) and method in self.methods:
            self.method = method
        else:
            raise TypeError(f'Bad method: {method}. Available methods: {", ".join(self.methods)}')

    def build_tree(self) -> BaseTree:
        """Construct the phylogenetic tree."""
        if self.method == "upgma":
            tree = self.upgma()
        else:
            tree = self.nj()
        return tree

    def upgma(self) -> BaseTree:
        """Construct an UPGMA tree."""
        # make a copy of the distance matrix to be used
        dm = self.dm
        # init terminal clades
        clades = [BaseTree.Clade(None, name) for name in dm.names]
        # init minimum index
        min_i = 0
        min_j = 0
        inner_count = 0
        pbar = tqdm(total=len(dm), desc='Assigning neighbours', smoothing=0.1)
        while len(dm) > 1:
            # find minimum
            min_i, min_j = np.unravel_index(dm.matrix.argmin(), dm.matrix.shape)
            min_dist = dm.matrix[min_i, min_j]
            # create clade
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            inner_count += 1
            inner_clade = BaseTree.Clade(None, "Inner" + str(inner_count))
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            # assign branch length
            if clade1.is_terminal():
                clade1.branch_length = min_dist * 1.0 / 2
            else:
                clade1.branch_length = min_dist * 1.0 / 2 - self._height_of(clade1)

            if clade2.is_terminal():
                clade2.branch_length = min_dist * 1.0 / 2
            else:
                clade2.branch_length = min_dist * 1.0 / 2 - self._height_of(clade2)
            # update node list
            clades[min_j] = inner_clade
            del clades[min_i]
            # rebuild distance matrix,
            # set the distances of new node at the index of min_j
            for k in trange(0, len(dm), desc='Rebuild distance matrix', leave=False):
                if k != min_i and k != min_j:
                    dm[min_j, k] = (dm[min_i, k] + dm[min_j, k]) * 1.0 / 2
            dm.names[min_j] = "Inner" + str(inner_count)
            pbar.update()
            del dm[min_i]
        inner_clade.branch_length = 0
        return BaseTree.Tree(inner_clade)

    def nj(self) -> BaseTree:
        """Construct a Neighbor-Joining tree."""
        # make a copy of the distance matrix to be used
        dm = copy.deepcopy(self.dm)
        # init terminal clades
        clades = [BaseTree.Clade(None, name) for name in dm.names]
        # init node distance
        node_dist = [0] * len(dm)
        # init minimum index
        min_i = 0
        min_j = 0
        inner_count = 0
        # special cases for Minimum Alignment Matrices
        if len(dm) == 1:
            root = clades[0]
            return BaseTree.Tree(root, rooted=False)
        elif len(dm) == 2:
            # minimum distance will always be [1,0]
            min_i = 1
            min_j = 0
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            clade1.branch_length = float(dm[min_i, min_j] / 2.0)
            clade2.branch_length = float(dm[min_i, min_j] - clade1.branch_length)
            inner_clade = BaseTree.Clade(None, "Inner")
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            clades[0] = inner_clade
            root = clades[0]
            return BaseTree.Tree(root, rooted=False)
        pbar = tqdm(total=len(dm) - 2, desc='Assigning neighbours', smoothing=0.1)
        while len(dm) > 2:
            # calculate nodeDist
            node_dist = dm.matrix.sum(0) / (len(dm) - 2)
            # find minimum distance pair
            q = dm.matrix.copy() - node_dist  # apply row-wise
            q = (q.T - node_dist).T  # apply column-wise
            np.fill_diagonal(q, np.inf)  # remove diagonal from solutions
            min_i, min_j = np.unravel_index(q.argmin(), q.shape)
            min_dist = q[min_i, min_j]
            # create clade
            clade1 = clades[min_i]
            clade2 = clades[min_j]
            inner_count += 1
            inner_clade = BaseTree.Clade(None, "Inner" + str(inner_count))
            inner_clade.clades.append(clade1)
            inner_clade.clades.append(clade2)
            # assign branch length
            clade1.branch_length = float(
                dm.matrix[min_i, min_j] + node_dist[min_i] - node_dist[min_j]
            ) / 2.0
            clade2.branch_length = float(dm.matrix[min_i, min_j] - clade1.branch_length)
            # update node list
            clades[min_j] = inner_clade
            del clades[min_i]
            # rebuild distance matrix,
            # set the distances of new node at the index of min_j
            dm.matrix[min_j, :] = dm.matrix[:, min_j] = (dm.matrix[min_j, :] + dm.matrix[min_i, :] - dm.matrix[min_i, min_j]) / 2
            np.fill_diagonal(dm.matrix, 0)
            dm.names[min_j] = "Inner" + str(inner_count)
            pbar.update()
            del dm[int(min_i)]
        # set the last clade as one of the child of the inner_clade
        root = None
        if clades[0] == inner_clade:
            clades[0].branch_length = 0
            clades[1].branch_length = float(dm[1, 0])
            clades[0].clades.append(clades[1])
            root = clades[0]
        else:
            clades[0].branch_length = float(dm[1, 0])
            clades[1].branch_length = 0
            clades[1].clades.append(clades[0])
            root = clades[1]
        return BaseTree.Tree(root, rooted=False)

    def _height_of(self, clade):
        """Calculate clade height -- the longest path to any terminal."""
        height = 0
        if clade.is_terminal():
            height = clade.branch_length
        else:
            height = height + max(self._height_of(c) for c in clade.clades)
        return height


def get_protein_mapping(protein_data: pd.DataFrame,
                        classes: Union[dict, List[dict]] = [{'l2': 'Kinase'}, {'l5': 'Adenosine receptor'}],
                        colors: List[str] = ['red', 'blue'],
                        generic_regex: bool = False) -> pd.DataFrame:
    """Create color mappings for specified protein classes

    :param protein_data: the dataframe of Papyrus protein targets
    :param classes: protein classes to keep (case insensitive).
                    - {'l2': 'Kinase'} matches all proteins with classification 'Enzyme->Kinase'
                    - {'l5': 'Adenosine receptor'} matches 'Membrane receptor->Family A G protein-coupled receptor->Small molecule receptor (family A GPCR)->Nucleotide-like receptor (family A GPCR)-> Adenosine receptor'
                    - All levels in the same dict are enforced, e.g. {'l1': ''Epigenetic regulator', 'l3': 'HDAC class IIb'} does not match records without the specified l1 AND l3
                    - If given a list of dicts, results in a union of the dicts, e.g. [{'l2': 'Kinase'}, {'l1': 'Membrane receptor'}] matches records with classification either 'Enzyme->Kinase' or 'Membrane receptor'
                    - Level-independent patterns can be specified with the 'l?' key, e.g. {'l?': 'SLC'} matches any classification level containing the 'SLC' keyword
                      Only one 'l?' per dict is supported.
                      Mixed usage of 'l?' and level-specific patterns (e.f. 'l1') is not supported
    :param colors: list of colors to be mapped to the provided classes
    :param generic_regex: whether to consider generic patterns 'l?' as regex, allowing for partial match.

    :return: the color data of desired protein classes
    """
    if isinstance(classes, dict):
        classes = [classes]
    # Verify classification keys
    keys = set(key for keys in classes for key in keys.keys())
    allowed_keys = ['l?', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8']
    if keys.difference(allowed_keys):
        raise ValueError(f'levels of protein classes must be of {allowed_keys}')
    if len(classes) != len(colors):
        raise ValueError(f'Number of colors ({len(colors)}) must correspond to number of classes ({len(classes)})')
    for key in classes:
        if 'l?' in key.keys():
            if len(key.keys()) > 1:
                raise ValueError(f'only one pattern per "l?" is accepted')
    # Split classifications
    ## 1) Handle multiple classifications
    split_classes = protein_data['Classification'].str.split(';')
    split_classes = equalize_cell_size_in_column(split_classes, 'external', '')
    split_classes = pd.DataFrame(split_classes.tolist())
    ## 2) Split into classification levels
    multiplicity = len(split_classes.columns)  # Number of max classifications
    for j in range(multiplicity):
        split_classes.iloc[:, j] = split_classes.iloc[:, j].str.split('->')
        split_classes.iloc[:, j] = equalize_cell_size_in_column(split_classes.iloc[:, j], 'external', '')
        # Ensure 8 levels of classification
        for _ in range(8 - len(split_classes.iloc[0, j])):
            split_classes.iloc[0, j].append('')
        split_classes.iloc[:, j] = equalize_cell_size_in_column(split_classes.iloc[:, j])
    ## 3) Create DataFrame with all annotations
    split_classes = pd.concat(
        [pd.DataFrame(split_classes.iloc[:, j].tolist(), columns=[f'l{x + 1}_{j + 1}' for x in range(8)]) for j in
         range(multiplicity)], axis=1)
    # Ensure case insensitivity
    split_classes = split_classes.apply(lambda s: s.str.lower())
    # Filter classes
    filtered = []
    for i, key in enumerate(classes):
        ## 1) Deal with specific protein classes (i.e. l1 to l8)
        if 'l?' not in key.keys():
            # Handle proteins of multiple classes
            for k in range(multiplicity):
                # Ensure all levels match
                query = '(' + ' and '.join([f'`{subkey.lower()}_{k + 1}` == "{subval.lower()}"'
                                         for subkey, subval in key.items()]) + ')'
                subfiltered = split_classes.query(query).index
                if not subfiltered.empty:
                    subfiltered = protein_data.iloc[subfiltered, :]
                    # Add annotation
                    subfiltered['Class'] = '->'.join(subval.lower() for _, subval in key.items())
                    subfiltered['Color'] = colors[i]
                    filtered.append(subfiltered)
        else:
            # Handle proteins of multiple classes
            for k in range(multiplicity):
                if generic_regex:  # match with Regex
                    # Match the concerned subset of columns
                    for column in split_classes.columns.str.endswith(f'_{k + 1}'):
                        subfiltered = split_classes[split_classes[column].str.lower().str.contains(list(key.items())[0][1].lower(), regex=True)]
                        if not subfiltered.empty:
                            subfiltered = protein_data.iloc[subfiltered, :]
                            subfiltered['Class'] = '->'.join(subval.lower() for _, subval in key.items())
                            subfiltered['Color'] = colors[i]
                            filtered.append(subfiltered)
                else:  # No regex pattern
                    # Match any level with the specified value
                    query = '(' + ' or '.join([f'`{column}` == "{list(key.items())[0][1].lower()}"'
                                         for column in split_classes.columns.str.endswith(f'_{k + 1}')]) + ')'
                    subfiltered = split_classes.query(query).index
                    if not subfiltered.empty:
                        subfiltered = protein_data.iloc[subfiltered, :]
                        subfiltered['Class'] = '->'.join(subval.lower() for _, subval in key.items())
                        subfiltered['Color'] = colors[i]
                        filtered.append(subfiltered)
    # Obtain targets from filtered indices
    return pd.concat(filtered, axis=0)


def get_closest(ref_x: float, ref_y: float, coordinates: Iterable[Tuple[float, float]]) -> Tuple[int, float]:
    """Identify the point closest to the reference

    :param ref_x: x coordinate of the reference point
    :param ref_y: y coordinate of the reference point
    :param coordinates: coordinates of other points
    :return: index of closest point in coordinates and distance to the reference point
    """
    distance = lambda x1, y1, x2, y2: ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    min_i, min_dist = float('inf'), float('inf')
    for i, (x, y) in enumerate(coordinates):
        dist = distance(ref_x, ref_y, x, y)
        if dist < min_dist:
            min_i, min_dist = i, dist
    return min_i, min_dist


def process_figtree_svg(svg_path: str,
                        out_path: str,
                        name_mapping: Dict[str, str] = None,
                        scaling: float=1.0) -> None:
    """Transform a FigTree SVG file to allow coloring of circle tip shapes according to mapping.

    :param svg_path: path to SCG FigTree phylogenetic tree to be transformed
    :param out_path: path to SVG file to be written to disk
    :param name_mapping: mapping between branch tips and colors
    :param scaling: scaling factor for radii of circles
    """
    if not os.path.isfile(svg_path):
        raise FileNotFoundError(f'SVG file does not exist: {svg_path}')
    # Read SVG as XML
    tree = etree.parse(svg_path)
    rotation_pattern = re.compile(r'translate\([-0-9.]+,[-0-9.]+\)\s+'
                                  r'matrix\([-0-9.]+,[-0-9.]+,[-0-9.]+,[-0-9.]+,([-0-9.]+),([-0-9.]+)\)|'
                                  r'matrix\([-0-9.]+,[-0-9.]+,[-0-9.]+,[-0-9.]+,([-0-9.]+),([-0-9.]+)\)')
    circle_pattern = re.compile(r'M([0-9.]+)\s([0-9.]+)\s'
                                r'C[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s'
                                r'C[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s([0-9.]+)\s([0-9.]+)\s'
                                r'C[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s'
                                r'C[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s[0-9.]+\s'
                                r'Z')
    # Get mapping of tip labels and their position
    tip_labels = {}
    # Iterate over text fields
    for node in tree.xpath("/*[name()='svg']/*[name()='g']/*[name()='g']/*[name()='text']"):
        # Identify rotated text
        match = re.match(rotation_pattern, node.getparent().attrib['transform'])
        if not match:
            continue
        label = node.text
        x, y = (float(x) for x in match.groups() if x is not None)
        tip_labels[label] = (x, y)
        # Remove tip label from SVG
        node.getparent().remove(node)


    # Iterate over paths
    for node in tree.xpath("/*[name()='svg']/*[name()='g']/*[name()='g']/*[name()='path']"):
        # Identify paths to transform into circles
        match = re.match(circle_pattern, node.attrib['d'])
        if not match:
            continue
        x1, y1, x2, y2 = (float(x) for x in match.groups())
        # Determine center and radius of circle from Bezier curve control points
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        radius = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 2 * scaling
        # Copy other attributes but path
        node_attribs = node.attrib
        del node_attribs['d']
        # Get color to attribute
        if name_mapping is not None:
            i, _ = get_closest(center_x, center_x, tip_labels.values())
            if i == float('inf'):
                node.getparent().remove(node)
                continue
            node_label = list(tip_labels.keys())[i]
            # Remove from tip labels
            del tip_labels[node_label]
            node_color = name_mapping.get(node_label, None)
            # Create XML element
            if node_color is None:
                # No coloring, remove tip shape
                node.getparent().remove(node)
            else:
                # Create colored circle
                circle = etree.Element('circle', cx=f"{center_x}", cy=f"{center_y}", r=f"{radius}",
                                       style=f"stroke:none; fill:{node_color};")
                # Replace path by circle
                node.getparent().replace(node, circle)
        else:
            # Create circle
            circle = etree.Element('circle', cx=f"{center_x}", cy=f"{center_y}", r=f"{radius}", style="stroke:none;")
            # Replace path by circle
            node.getparent().replace(node, circle)
    # Write transformed SVG to disk
    tree.write(out_path, pretty_print=True)


def create_phylotree(root_dir: str,
                     out_dir: str,
                     version: str,
                     kalign_path: str,
                     njobs: int):
    # Output file names
    processed_fasta = os.path.join(out_dir, 'Human_protein_targets.fasta')
    aligned_fasta = os.path.join(out_dir, 'Human_protein_targets_Kalign_alignment.fasta')
    distance_matrix = os.path.join(out_dir, 'Human_protein_targets_dense_distance_matrix_Kalign.npy')
    sequence_names =  os.path.join(out_dir, 'Human_protein_targets_dense_matrix_names_Kalign.txt.xz')
    phylo_tree_newick = os.path.join(out_dir, 'Human_protein_targets_Kalign_nj_tree.nwk')
    # Read protein data and keep only human targets
    protein_data = read_protein_set(source_path=root_dir, version=version)
    protein_data = protein_data[protein_data.Organism.str.startswith('Homo sapiens')]
    dataframe_to_fasta(protein_data, processed_fasta)
    # Align sequences using Kalign
    run_kalign(processed_fasta, aligned_fasta, kalign_path)
    # Obtain distance matrix using Biopython
    aln = AlignIO.read(aligned_fasta, 'fasta')
    # Number of calculation to be performed
    n = sum(1 for _ in combinations(aln, 2))
    print(f'Number of calculations to be performed: {n}')
    n_logic_cores = psutil.cpu_count(logical=True)
    print(f'Number of available logical cores: {n_logic_cores}')
    print(f'Number of cores to be used: {njobs}')
    # Run parallel distance calculations
    pbar = tqdm(enumerate(combinations(aln, 2)), total=n, desc='Calculating distances')
    distances = Parallel(n_jobs=njobs, verbose=0)(delayed(get_distance_from_aligned_seqs)(seq1, seq2, 'BLOSUM62')
                                               for i, (seq1, seq2) in pbar)
    # Arrange distance in symmetric distance matrix
    names = {al_seq.description: i for i, al_seq in enumerate(aln)}
    matrix = np.zeros((len(names.keys()), len(names.keys())), dtype=np.float64)
    for id1_name, id2_name, dist in tqdm(distances):
        id1, id2 = names[id1_name], names[id2_name]
        matrix[id1, id2] = dist
    # Save distance matrix and names to disk
    np.save(distance_matrix, matrix)
    names = list(names.keys())
    with lzma.open(sequence_names, 'wt') as oh:
        oh.write('\n'.join(names))
    # Build phylogenetic tree
    dm = DistanceMatrix(names, matrix)
    dtc = DistanceTreeConstructor(dm, 'nj')
    tree = dtc.build_tree()
    # Allow deeper recursion calls to write tree to disk
    sys.setrecursionlimit(100000)
    Phylo.write(tree, phylo_tree_newick, 'newick')
    print('The phylogenetic tree was written to disk.\n'
          'To be post-processed with the \'process\' command:\n'
          '    1) Open the Newick tree with FigTree,\n'
          '       (https://github.com/rambaut/figtree/releases)\n'
          '    2) Display the tree using a radial layout,\n'
          '    3) Under \'Trees\':\n'
          '        - Root the tree using \'midpoint\',\n'
          '        - Transform branches to \'cladogram\',\n'
          '    4) Tick \'Tip Shapes\',\n'
          '    5) Untick \'Scale Bar\',\n'
          '    6) Under \'File > Export SVG...\' save the obtained figure,\n'
          '    7) Process the saved file with the \'process\' command\n')


class hashdict(dict):
    """A hashable dict to be used as a dict's keys."""
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def process_phylotree(infile: str,
                      outfile: str,
                      root_dir: str,
                      version: str,
                      highlights: str,
                      scaling: float):
    if not os.path.isfile(os.path.abspath(infile)):
        raise ValueError(f'input file could not be found: {infile}')
    # Identify the proteins to highlight
    protein_data = read_protein_set(source_path=root_dir, version=version)
    protein_data = protein_data[protein_data.Organism.str.startswith('Homo sapiens')]
    if highlights is not None:
        try:
            # Identify dict encoded as str
            # and replace by hashdict
            patt = re.compile("({(?:(?:(?:'[^'\\\\]*(?:\\\\.[^'\\\\]*)*')|(?:\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\"))[: ,]*)+})")
            highlights = re.sub(patt, 'hashdict(\\1)', highlights)
            # Parse highlight str values
            highlights = eval(highlights)
        except:
            raise ValueError('highlights must be key-value pairs like {classX: colorY}')
        # Cast appropriately
        protein_class = list(highlights.keys())
        color = list(highlights.values())
        # Obtain the name of the proteins of interest
        mapping_data = get_protein_mapping(protein_data,
                                           classes=protein_class,
                                           colors=color)
        mappings = dict(mapping_data[['target_id', 'Color']].values)
    else:
        mappings = {}
    # Write processed SVG file
    process_figtree_svg(infile, outfile, mappings, scaling)


class _HelpAction(argparse._HelpAction):
    '''Custom help formatter for argparse.
    
    Adapted from: https://stackoverflow.com/a/24122778
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        # Retrieve subparsers from parser
        subparsers_actions = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)]
        # Print help for all subparsers
        for subparsers_action in subparsers_actions:
            for choice, subparser in subparsers_action.choices.items():
                print('================================')
                print(f'\n   SUBCOMMAND {choice}\n\n')
                print(subparser.format_help())
        parser.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and post-process phylogenetic trees of the Papyrus data.',
                                     add_help=False)
    parser.add_argument('-h', '--help',
                        action=_HelpAction,
                        help='show this help message and exit')
    subparsers = parser.add_subparsers(required=True, dest='command')
    # Add switches independent of subcommands
    # Parser for tree creation
    creation_parser = subparsers.add_parser('create',
                                            description='Create the phylogenetic tree',
                                            formatter_class=argparse.RawTextHelpFormatter)
    creation_parser.add_argument('-i', '--indir',
                                 required=False,
                                 default=None,
                                 help=('Directory containing Papyrus\'s data directory\n'
                                       '(default: pystow\'s home folder).'),
                                 dest='indir')
    creation_parser.add_argument('-o', '--outdir',
                                 default='./',
                                 required=False,
                                 help=('Directory where phylogenetic trees will be created\n'
                                       '(default: current folder).'),
                                 dest='outdir')
    creation_parser.add_argument('-V', '--version',
                                 default='latest',
                                 required=False,
                                 help=('Version of the Papyrus data to be used (default: latest).'),
                                 dest='version')
    creation_parser.add_argument('-k', '--kalign_path',
                                 required=False,
                                 default='kalign',
                                 help=('Path to the kalign executable\n'
                                       '(default: assumes kalign is in the PATH environment variable).'),
                                 dest='kalign')
    creation_parser.add_argument('--njobs',
                                 type=int,
                                 default=-2,
                                 required=False,
                                 help=('Number of concurrent processes for distance calculation\n'
                                       '(default: #cores - 1).'),
                                 dest='njobs')
    # Parser for the postprocessing of FigTree SVG files
    process_parser = subparsers.add_parser('process',
                                           description='Process the FigTree SVG tree file',
                                           formatter_class=argparse.RawTextHelpFormatter)
    process_parser.add_argument('-f', '--infile',
                                required=True,
                                help=('SVG file generated by FigTree.'),
                                dest='infile')
    process_parser.add_argument('-o', '--outfile',
                                required=True,
                                help=('Post-processed SVG file to be written to disk.'),
                                dest='outfile')
    process_parser.add_argument('-i', '--indir',
                                required=False,
                                default=None,
                                help=('Directory containing Papyrus\'s data directory\n'
                                      '(default: pystow\'s home folder).'),
                                dest='indir')
    creation_parser.add_argument('-V', '--version',
                                 default='latest',
                                 required=False,
                                 help=('Version of the Papyrus data to be used (default: latest).'),
                                 dest='version')
    process_parser.add_argument('-H', '--highlights',
                                default=None,
                                required=False,
                                help=('Protein families to highlight.\n'
                                      'Must correspond to a dictionary with:\n'
                                      '    - key elements  being the (case insensitive) protein class\n'
                                      '      (either a unique dictionary or a list of dictionaries):\n'
                                      '        - {\'l2\': \'Kinase\'} matches all proteins\n'
                                      '          with classification \'Enzyme->Kinase\'\n'
                                      '        - {\'l5\': \'Adenosine receptor\'} matches\n'
                                      '          \'Membrane receptor->Family A G protein-coupled receptor->\n'
                                      '             Small molecule receptor (family A GPCR)->\n'
                                      '             Nucleotide-like receptor (family A GPCR)->\n'
                                      '             Adenosine receptor\'\n'
                                      '        - All levels in the same dict are enforced,\n'
                                      '          e.g. {\'l1\': \'Epigenetic regulator\',\n'
                                      '                \'l3\': \'HDAC class IIb\'}\n'
                                      '          does not match records without the specified l1 AND l3.\n'
                                      '        - If given a list of dicts, results in a union of the dicts,\n'
                                      '          e.g. [{\'l2\': \'Kinase\'}, {\'l1\': \'Membrane receptor\'}]\n'
                                      '          matches records with classification either\n'
                                      '          \'Enzyme->Kinase\' or \'Membrane receptor\'.\n'
                                      '        - Level-independent patterns can be specified with the \'l?\' key,\n'
                                      '          e.g. {\'l?\': \'SLC\'} matches any classification level\n'
                                      '          containing the \'SLC\' keyword.\n'
                                      '        - Only one \'l?\' per dict is supported.\n'
                                      '        - Mixed usage of \'l?\' and level-specific patterns\n'
                                      '          (e.g. \'l1\') is not supported in the same dictionary.\n'
                                      '    - value elements the color to associate in hexadecimal.\n\n'
                                      'e.g. {[{\'l2\': \'Kinase\'}, {\'l1\': \'Membrane receptor\'}]: \'#0072B2\'}\n'
                                      '     {{\'l?\': \'SLC\'}: \'#CC79A7\', {\'l3\': \'Serine protease\'}: \'#D67AD2\'}\n\n'),
                                dest='highlights')
    process_parser.add_argument('--scaling',
                                required=False,
                                default=1.0,
                                type=float,
                                help=('Factor to scale the circles by.\n'),
                                dest='scaling')
    args = parser.parse_args()
    if args.command == 'create':
        # Create phylo output folder
        outdir = os.path.join(os.path.abspath(args.outdir), 'phylo')
        os.makedirs(outdir, exist_ok=True)
        create_phylotree(root_dir=args.indir,
                         out_dir=outdir,
                         version=args.version,
                         kalign_path=args.kalign,
                         njobs=args.njobs)
    elif args.command == 'process':
        process_phylotree(infile=args.infile,
                          outfile=args.outfile,
                          root_dir=args.indir,
                          version=args.version,
                          highlights=args.highlights,
                          scaling=args.scaling)
