# -*- coding: utf-8 -*-

import lzma
import os
import pickle
import argparse
from collections import Counter
from typing import List, Union

import pystow
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm, trange

from papyrus_scripts.utils.IO import process_data_version

# Required dependencies
try:
    import tmap as tm
except ImportError as e:
    tm = e
try:
    from faerun import Faerun
except ImportError as e:
    Faerun = e
try:
    from mhfp.encoder import MHFPEncoder
except ImportError as e:
    MHFPEncoder = e


# Handle missing librairies
missing = []
for name, dependency in [('tmap (conda)', tm),
                         ('faerun', Faerun),
                         ('mhfp', MHFPEncoder)]:
    if isinstance(dependency, ImportError):
        missing.append(name)
if len(missing):
    raise ImportError(f'Some required dependencies are missing:\n\t{", ".join(missing)}')


def get_mol_props_for_map(xzfile):
    """Return the heavy atom count, carbon fraction, ring atom fraction and largest ring size."""
    hac = []
    c_frac = []
    ring_atom_frac = []
    largest_ring_size = []

    with lzma.open(xzfile) as fh, Chem.ForwardSDMolSupplier(fh) as supplier:
        for i, mol in enumerate(tqdm(supplier, total=1268606, ncols=90, desc='Obtaining molecular properties')):
            atoms = mol.GetAtoms()
            size = mol.GetNumHeavyAtoms()
            n_c = 0
            n_ring_atoms = 0
            for atom in atoms:
                if atom.IsInRing():
                    n_ring_atoms += 1
                if atom.GetSymbol().lower() == "c":
                    n_c += 1

            c_frac.append(n_c / size)
            ring_atom_frac.append(n_ring_atoms / size)
            sssr = AllChem.GetSymmSSSR(mol)
            if len(sssr) > 0:
                largest_ring_size.append(max([len(s) for s in sssr]))
            else:
                largest_ring_size.append(0)
            hac.append(size)
    return hac, c_frac, ring_atom_frac, largest_ring_size


def ordered_top_labels(series: Union[pd.Series, List], top_n: int):
    """Return the Faerun labels and data of the top occurences, others categorized as 'Other'.

    :param series: the data
    :param top_n: the number of categories other than 'Other' to keep
    """
    labels, data = Faerun.create_categories(series)
    top = [i for i, _ in Counter(data).most_common(top_n)]

    top_labels = [(7, "Other")]
    map_ = [7] * len(data)
    value = 1
    for i, name in labels:
        if i in top:
            v = value
            if v == 7:
                v = 0
            top_labels.append((v, name))
            map_[i] = v
            value += 1
    data = [map_[val] for _, val in enumerate(data)]
    return labels, data


def get_mols(xzfile):
    """Return the heavy atom count, carbon fraction, ring atom fraction and largest ring size."""
    with lzma.open(xzfile) as fh, Chem.ForwardSDMolSupplier(fh) as supplier:
        for mol in supplier:
            yield mol
    return


def get_sources(tsv_file):
    """Organize the data by source."""
    # Identify if stereo is included or not
    temp = pd.read_csv(tsv_file, sep='\t', low_memory=True, nrows=10)
    identifier = 'connectivity' if 'connectivity' in temp.columns else 'InChIKey'
    del temp
    # Obtain relevant information
    data = pd.concat([x for x in tqdm(pd.read_csv(tsv_file, sep='\t', low_memory=True, chunksize=1000000, usecols=[identifier, 'source', 'SMILES']), total=60, ncols=120)], axis=0)
    # Group data by connectivity/InChIKey
    grouped = data.groupby(identifier)
    # Determine unique sources for each molecule
    listvals = lambda x: ';'.join(set(str(y) for y in x)) if (x.values[0] == x.values).all() else ';'.join(str(y) for y in x)
    grouped2 = grouped.aggregate({'source': listvals, 'SMILES': listvals}).reset_index()
    # Reorder sources
    grouped2.loc[:, 'source'] = grouped2['source'].str.split(';').apply(lambda x: ';'.join(sorted(set(x))))
    # Get unique datasets
    unique_excape = grouped2.loc[~grouped2['source'].str.contains(';') & grouped2['source'].str.contains('ExCAPE'), [identifier, 'source']]
    unique_chembl = grouped2.loc[~grouped2['source'].str.contains(';') & grouped2['source'].str.contains('ChEMBL'), [identifier, 'source']]
    unique_sharma = grouped2.loc[~grouped2['source'].str.contains(';') & grouped2['source'].str.contains('Sharma'), [identifier, 'source']]
    unique_christ = grouped2.loc[~grouped2['source'].str.contains(';') & grouped2['source'].str.contains('Christman'), [identifier, 'source']]
    unique_merget = grouped2.loc[~grouped2['source'].str.contains(';') & grouped2['source'].str.contains('Merget'), [identifier, 'source']]
    # Get overlaps
    overlap_chembl_excape = grouped2.loc[(grouped2['source'].str.count(';') > 0) & grouped2['source'].str.contains('ExCAPE') & grouped2['source'].str.contains('ChEMBL') , [identifier, 'source']]
    overlaps_chembl = grouped2.loc[grouped2['source'].str.contains(';') & grouped2['source'].str.contains('ChEMBL') & ~grouped2['source'].str.contains('ExCAPE'), [identifier, 'source']]
    overlaps_others = grouped2.loc[grouped2['source'].str.contains(';') & ~grouped2['source'].str.contains('ChEMBL'), [identifier, 'source']]
    # Rename sources
    unique_excape.loc[:, 'source'] = 'ExCAPE-DB only'
    unique_chembl.loc[:, 'source'] = 'ChEMBL29 only'
    unique_sharma.loc[:, 'source'] = 'Sharma only'
    unique_christ.loc[:, 'source'] = 'Christmann-Franck only'
    unique_merget.loc[:, 'source'] = 'Merget only'
    overlap_chembl_excape.loc[:, 'source'] = 'ChEMBL29 & ExCAPE-DB'
    overlaps_chembl.loc[:, 'source'] = 'All but ExCAPE-DB'
    overlaps_others.loc[:, 'source'] = 'All but ChEMBL29'
    # Obtain dicts
    unique_excape = dict(unique_excape.values.tolist())
    unique_chembl = dict(unique_chembl.values.tolist())
    unique_sharma = dict(unique_sharma.values.tolist())
    unique_christ = dict(unique_christ.values.tolist())
    unique_merget = dict(unique_merget.values.tolist())
    overlap_chembl_excape = dict(overlap_chembl_excape.values.tolist())
    overlaps_chembl = dict(overlaps_chembl.values.tolist())
    overlaps_others = dict(overlaps_others.values.tolist())
    # Assemble
    sources = {**unique_excape, **unique_chembl, **unique_sharma, **unique_christ, **unique_merget, **overlap_chembl_excape, **overlaps_chembl, **overlaps_others}
    return sources


def get_tmap_MHFP(root_dir, struct_dir, out_dir):
    # Output files
    out_dat = os.path.join(out_dir, 'Papyrus_lshforest_without_stereo_MHFP.dat')
    label_file = os.path.join(out_dir, 'Papyrus_tmap_mol_labels.pickle')
    source_file = os.path.join(out_dir, 'Papyrus_tmap_mol_sources.pickle')
    mol_prop_file = os.path.join(out_dir, 'Papyrus_tmap_mol_props.pickle')
    out_plot = os.path.join(out_dir, 'Papyrus_TMAP_MHFP')
    coordinate_file = os.path.join(out_dir, 'Papyrus_MHFP_coordinates.dat')
    # Input file of activities
    activity_file = os.path.join(root_dir, '05.4_combined_set_without_stereochemistry.tsv.xz')
    # Input file of structures
    struct_file = os.path.join(struct_dir, '05.4_combined_2D_set_without_stereochemistry.sd.xz')

    # LSH forest encoder
    lf = tm.LSHForest(1024, 128, 8)
    # Encode molecules into MHFP
    if not os.path.isfile(out_dat):
        enc = MHFPEncoder(1024)
        fps = []
        for mol in tqdm(get_mols(struct_file), total=1268606, ncols=90, dec='Encoding molecules'):
            fps.append(tm.VectorUint(enc.encode_mol(mol)))
        lf.batch_add(fps)
        print('Added fingerprints to LSH')
        lf.index()
        print('Indexed LSH')
        lf.store(out_dat)
        print('Stored LSH')
        del fps
    else:
        # Restore LSH from disk
        print('Loading LSH from file')
        lf.restore(out_dat)
        print('Loaded LSH from file')
    # Create labels for molecules
    if not os.path.isfile(label_file):
        labels = []
        for mol in tqdm(get_mols(struct_file), total=1268606, ncols=90, desc='Determining labels'):
            smiles = Chem.MolToSmiles(mol)
            connectivity = mol.GetPropsAsDict().get('connectivity', '')
            labels.append(smiles + "__" + connectivity + '__' + smiles)
        with open(label_file, "wb+") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load from file
        print('Loading labels from file')
        with open(label_file, "rb") as fh:
            labels = pickle.load(fh)
        print('Loaded labels from file')
    # Obtain molecule sources
    if not os.path.isfile(source_file):
        unordered_sources = get_sources(activity_file)
        sources = []
        for mol in tqdm(get_mols(struct_file), total=1268606, ncols=90, desc='Determining sources'):
            connectivity = mol.GetPropsAsDict().get('connectivity', '')
            sources.append(unordered_sources[connectivity])
        with open(source_file, "wb+") as f:
            pickle.dump(sources, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load from file
        print('Loading molecular sources from file')
        with open(source_file, "rb") as f:
            sources = pickle.load(f)
        print('Loaded sources from file')
    # Calculate molecule properties for legend
    if not os.path.isfile(mol_prop_file):
        hac, c_frac, ring_atom_frac, ring_size = get_mol_props_for_map(struct_file)
        with open(mol_prop_file, "wb+") as f:
            pickle.dump(
                (hac, c_frac, ring_atom_frac, ring_size), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load from file
        print('Loading molecular properties')
        with open(mol_prop_file, "rb") as fh:
            hac, c_frac, ring_atom_frac, ring_size = pickle.load(fh)
        print('Loaded molecular properties')
    # Configure visulaization
    if not os.path.isfile(coordinate_file):
        cfg = tm.LayoutConfiguration()
        cfg.node_size = 1 / 70
        cfg.mmm_repeats = 2
        cfg.sl_repeats = 2
        cfg.sl_extra_scaling_steps = 10
        cfg.k = 50
        cfg.kc = 50
        cfg.fme_iterations = 1000000
        print('Determining coordinates and minimum spanning tree')
        x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
        x = list(x)
        y = list(y)
        s = list(s)
        t = list(t)
        print('Saving coordinates to file')
        with open(coordinate_file, "wb+") as f:
            pickle.dump((x, y, s, t), f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved coordinates')
    else:
        print('Loading layout coordinates')
        with open(coordinate_file, "rb") as f:
            x, y, s, t = pickle.load(f)
        print('Loaded layout coordinates')
    # Creating legend for categorical label
    source_labels, source_data = Faerun.create_categories(sources)
    print('Source categories created')
    # Creating scatter plot
    f = Faerun(view="front", coords=False, clear_color='#ffffff', alpha_blending=True)
    print('Adding scatter plot')
    f.add_scatter("Papyrus_nostereo", {'x': x,
                                       'y': y,
                                       'c': [
                                           source_data,
                                           hac,
                                           c_frac,
                                           ring_atom_frac,
                                           ring_size],
                                       'labels': labels},
                  shader="smoothCircle",
                  point_scale=2.0,
                  max_point_size=20,
                  legend_labels=[source_labels],
                  categorical=[True, False, False, False, False],
                  colormap=["Accent", "rainbow", "rainbow", "rainbow", "rainbow"],
                  series_title=[
                      "Source",
                      "Heavy atom count",
                      "Carbon fraction",
                      "Ring atom fraction",
                      "Largest ring size"],
                  has_legend=True,
                  )
    print('Adding tree')
    f.add_tree('papyrus_nostereo', {"from": s, "to": t}, point_helper="Papyrus_nostereo", color="#222222")
    print('Saving tmap')
    f.plot(out_plot, template='smiles')


def get_tmap_UniRep(root_dir, desc_dir, out_dir):
    # Output files
    out_dat = os.path.join(out_dir, 'Papyrus_lshforest_without_stereo_UniRep.dat')
    prot_prop_file = os.path.join(out_dir, 'Papyrus_tmap_prot_props.pickle')
    out_plot = os.path.join(out_dir, 'Papyrus_TMAP_UniRep')
    # Input files
    seq_file = os.path.join(root_dir, '05.4_combined_set_protein_targets.tsv.xz')
    desc_file = os.path.join(desc_dir, '05.4_combined_prot_embeddings_unirep.tsv.xz')
    # LSH and Minhash encoders
    lf = tm.LSHForest(6660, 128, 8)
    enc = tm.Minhash()

    # Read protein descriptors
    if not os.path.isfile(out_dat):
        data = pd.read_csv(desc_file, sep='\t')
        for i in trange(data.shape[0], ncols=90):
            fp = tm.VectorFloat(data.iloc[i, 1:])
            lf.add(enc.from_weight_array(fp, method='I2CWS'))
        print('Added fingerprints to LSH')
        lf.index()
        print('Indexed LSH')
        lf.store(out_dat)
        print('Stored LSH')
    else:
        # Restore LSH from disk
        lf.restore(out_dat)
    if not os.path.isfile(prot_prop_file):
        prot_data = pd.read_csv(seq_file, sep='\t')
        # Organisms with > 20k activity values
        allowed_organisms = ["Homo sapiens (Human)", "Mus musculus (Mouse)", "Rattus norvegicus (Rat)",
                             "Escherichia coli (strain K12)", "Equus caballus (Horse)",
                             "Influenza A virus (A/WSN/1933(H1N1))", "Trypanosoma cruzi",
                             "Schistosoma mansoni (Blood fluke)", "Bacillus subtilis"]
        organisms = [organism if organism in allowed_organisms else 'Other' for organism in prot_data['Organism']]
        lengths = prot_data['Length']
        # Protein classification
        classif = prot_data[~prot_data['Classification'].isna()]['Classification'].str.split(';').apply(
            lambda x: x[0]).str.split('->')
        l1 = classif.apply(lambda x: x[0])
        l2 = classif.apply(lambda x: x[1] if len(x) > 1 else '')
        l3 = classif.apply(lambda x: x[2] if len(x) > 2 else '')
        l4 = classif.apply(lambda x: x[3] if len(x) > 3 else '')
        l5 = classif.apply(lambda x: x[4] if len(x) > 4 else '')
        l6 = classif.apply(lambda x: x[5] if len(x) > 5 else '')
        del classif
        with open(prot_prop_file, "wb+") as f:
            pickle.dump(
                (organisms, lengths, l1, l2, l3, l4, l5, l6),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print('Stored properties')
    else:
        with open(prot_prop_file, "rb") as f:
            organisms, lengths, l1, l2, l3, l4, l5, l6 = pickle.load(f)
        print('Loaded properties')

    organisms_labels, organisms_data = ordered_top_labels(organisms, 9)
    l1_labels, l1_data = ordered_top_labels(l1, 9)
    l2_labels, l2_data = ordered_top_labels(l2, 9)
    l3_labels, l3_data = ordered_top_labels(l3, 9)
    l4_labels, l4_data = ordered_top_labels(l4, 9)
    l5_labels, l5_data = ordered_top_labels(l5, 9)
    l6_labels, l6_data = ordered_top_labels(l6, 9)

    # Layout configuration
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 10
    cfg.mmm_repeats = 2
    cfg.sl_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.k = 50
    cfg.kc = 50

    print('Creating layout')
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
    f = Faerun(view="front", coords=False, clear_color='#ffffff', alpha_blending=True)
    print('Adding scatter plot')
    f.add_scatter("Papyrus_nostereo", {'x': x,
                                       'y': y,
                                       'c': [
                                           lengths,
                                           organisms_data,
                                           l1_data,
                                           l2_data,
                                           l3_data,
                                           l4_data,
                                           l5_data,
                                           l6_data
                                       ]}, shader="smoothCircle", point_scale=2.0,

                  max_point_size=50,
                  legend_labels=[organisms_labels,
                                 l1_labels,
                                 l2_labels,
                                 l3_labels,
                                 l4_labels,
                                 l5_labels,
                                 l6_labels],
                  categorical=[False, True, True, True, True, True, True, True],
                  colormap=["rainbow", "tab10", "tab10", "tab10", "tab10", "tab10", "tab10", "tab10"],
                  series_title=[
                      "Sequence length",
                      "Organism",
                      "Protein class level 1",
                      "Protein class level 2",
                      "Protein class level 3",
                      "Protein class level 4",
                      "Protein class level 5",
                      "Protein class level 6"],
                  has_legend=True,
                  )
    print('Adding tree')
    f.add_tree('papyrus_nostereo', {"from": s, "to": t}, point_helper="Papyrus_nostereo", color="#222222")
    print('Saving tmap')
    f.plot(out_plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TMAPs of the Papyrus data.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--indir',
                        required=False,
                        default=None,
                        help=('Directory where Papyrus bioactivity data is stored\n'
                              '(default: pystow\'s home folder).'),
                        dest='indir')
    parser.add_argument('-V', '--version',
                        default='latest',
                        required=False,
                        help=('Version of the Papyrus data to be used (default: latest).'),
                        dest='version')
    parser.add_argument('-o', '--outdir',
                        default='./',
                        required=False,
                        help=('Directory where TMAP outputs will be stored\n'
                              '(default: current folder).'),
                        dest='outdir')
    parser.add_argument('-P',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Should a TMAP be generated for target sequences.',
                        dest='proteintmap')
    args = parser.parse_args()
    version = process_data_version(args.version)
    if args.indir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(args.indir)
    papyrus_root = pystow.module('papyrus', version)
    structure_folder = papyrus_root.join('structures')
    descriptor_folder = papyrus_root.join('descriptors')
    # Create TMAP output folder
    outdir = os.path.join(os.path.abspath(args.outdir), 'tmap')
    os.makedirs(outdir, exist_ok=True)
    # Create TMAP of molecular structures
    get_tmap_MHFP(papyrus_root.base.as_posix(),
                  structure_folder,
                  outdir)
    # Create TMAP of protein sequences
    if args.proteintmap:
        get_tmap_UniRep(papyrus_root.base.as_posix(),
                        descriptor_folder,
                        outdir)
