# -*- coding: utf-8 -*-
import lzma
import os
import pickle
from collections import Counter
from typing import List, Union

import pandas as pd
import tmap as tm
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm, trange


def get_mol_props_for_map(xzfile):
    """Return the heavy atom count, carbon fraction, ring atom fraction and largest ring size."""
    hac = []
    c_frac = []
    ring_atom_frac = []
    largest_ring_size = []

    with lzma.open(xzfile) as fh, Chem.ForwardSDMolSupplier(fh) as supplier:
        for i, mol in enumerate(tqdm(supplier, total=1268606, ncols=90)):
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
    mols = []
    with lzma.open(xzfile) as fh, Chem.ForwardSDMolSupplier(fh) as supplier:
        for i, mol in enumerate(tqdm(supplier, total=1268606, ncols=90)):
            mols.append(mol)
    return mols


def get_tmap_MHFP(folder):
    # Output files
    out_file = "Papyrus_lshforest_without_stereo_MHFP.dat"
    label_file = 'mol_labels.pickle'
    mol_prop_file = "props.pickle"

    # LSH forest encoder
    lf = tm.LSHForest(1024, 128, 8)
    # Obtain molecules
    if not os.path.isfile('mol_labels.pickle') or not os.path.isfile(out_file):
        mols = get_mols(os.path.join(folder, '05.4_combined_2D_set_without_stereochemistry.sd.xz'))
    # Encode into MHFP
    if not os.path.isfile(out_file):
        enc = MHFPEncoder(1024)
        fps = []
        for mol in tqdm(mols, total=1268606, ncols=90):
            fps.append(tm.VectorUint(enc.encode_mol(mol)))
        lf.batch_add(fps)
        print('Added fingerprints to LSH')
        lf.index()
        print('Indexed LSH')
        lf.store(out_file)
        print('Stored LSH')
    else:
        # Restore LSH from disk
        lf.restore(out_file)
    # Create labels for molecules
    if not os.path.isfile(label_file):
        labels = []
        for mol in tqdm(mols, total=1268606, ncols=90):
            smiles = Chem.MolToSmiles(mol)
            connectivity = mol.GetPropsAsDict().get('connectivity', '')
            labels.append(smiles + "__" + connectivity + '__' + smiles)
        with open(label_file, "wb+") as f:
            pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load from file
        labels = pickle.load(label_file)
    # Calculate molecule properties for legend
    if not os.path.isfile(mol_prop_file):
        hac, c_frac, ring_atom_frac, ring_size = get_mol_props_for_map(
            os.path.join(folder, '05.4_combined_2D_set_without_stereochemistry.sd.xz'))
        with open(mol_prop_file, "wb+") as f:
            pickle.dump(
                (hac, c_frac, ring_atom_frac, ring_size), f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Stored properties')
    else:
        # Load from file
        with open(mol_prop_file, "rb") as f:
            hac, c_frac, ring_atom_frac, ring_size = pickle.load(f)
        print('Loaded properties')
    # Configure visulaization
    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 70
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
                                           hac,
                                           c_frac,
                                           ring_atom_frac,
                                           ring_size],
                                       'labels': labels},
                  shader="smoothCircle",
                  point_scale=2.0,
                  max_point_size=20,
                  categorical=[False, False, False, False],
                  colormap=["rainbow", "rainbow", "rainbow", "rainbow"],
                  series_title=[
                      "Heavy atom count",
                      "Carbon fraction",
                      "Ring atom fraction",
                      "Largest ring size"],
                  has_legend=True,
                  )
    print('Adding tree')
    f.add_tree('papyrus_nostereo', {"from": s, "to": t}, point_helper="Papyrus_nostereo", color="#222222")
    print('Saving tmap')
    f.plot("Papyrus_no_stereo_MHFP", template='smiles')


def get_tmap_UniRep(folder):
    out_file = "Papyrus_lshforest_without_stereo_UniRep.dat"
    prot_prop_file = "protprops.pickle"
    # LSH and Minhash encoders
    lf = tm.LSHForest(6660, 128, 8)
    enc = tm.Minhash()

    # Read protein descriptors
    if not os.path.isfile(out_file):
        data = pd.read_csv(os.path.join(folder, '05.4_combined_prot_embeddings_unirep.tsv.gz'), sep='\t')
        for i in trange(data.shape[0], ncols=90):
            fp = tm.VectorFloat(data.iloc[i, 1:])
            lf.add(enc.from_weight_array(fp, method='I2CWS'))
        print('Added fingerprints to LSH')
        lf.index()
        print('Indexed LSH')
        lf.store(out_file)
        print('Stored LSH')
    else:
        # Restore LSH from disk
        lf.restore(out_file)
    if not os.path.isfile(prot_prop_file):
        prot_data = pd.read_csv(os.path.join(folder, '05.4_combined_set_protein_targets.tsv.gz'), sep='\t')
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

    # Layoiut configuration
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
    f.plot("Papyrus_no_stereo_UniRep")

if __name__ == '__main__':
    # Modify path to folder containing Papyrus molecular and protein descriptors
    FOLDER = ''
    get_tmap_MHFP(FOLDER)
    get_tmap_UniRep(FOLDER)