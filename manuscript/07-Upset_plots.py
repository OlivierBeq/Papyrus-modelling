# -*- coding: utf-8 -*-

import os
import lzma
import argparse

import pystow
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt

from papyrus_scripts.utils.IO import process_data_version

try:
    from upsetplot import from_contents, UpSet
except ImportError as e:
    raise ImportError('Some required dependencies are missing:\n\thttps://github.com/OlivierBeq/UpSetPlot')


def compound_upset_plot(sd_xz, tsv_xz, plot_png, **kwargs):
    # Get order of connectivities
    connectivities = []
    with lzma.open(sd_xz) as fh, Chem.ForwardSDMolSupplier(fh) as supplier:
        for mol in tqdm(supplier, total=1268606):
            connectivities.append(mol.GetProp('connectivity'))
    connectivities = pd.DataFrame(connectivities, columns=['connectivity'])
    # Read sources and connectivities
    sources = pd.read_csv(tsv_xz, sep='\t', usecols=['connectivity', 'source'])
    # Get rows with corresponding sources and identify their indices in the set of molecules
    ChEMBL_connectivities = sources[sources.source.str.contains('ChEMBL29')].connectivity.unique()
    ChEMBL_indices = connectivities[connectivities.connectivity.isin(ChEMBL_connectivities)].index.tolist()
    ExCAPE_connectivities = sources[sources.source.str.contains('ExCAPE-DB')].connectivity.unique()
    ExCAPE_indices = connectivities[connectivities.connectivity.isin(ExCAPE_connectivities)].index.tolist()
    Sharma_connectivities = sources[sources.source.str.contains('Sharma2016')].connectivity.unique()
    Sharma_indices = connectivities[connectivities.connectivity.isin(Sharma_connectivities)].index.tolist()
    Klaeger_connectivities = sources[sources.source.str.contains('Klaeger2017')].connectivity.unique()
    Klaeger_indices = connectivities[connectivities.connectivity.isin(Klaeger_connectivities)].index.tolist()
    Merget_connectivities = sources[sources.source.str.contains('Merget2017')].connectivity.unique()
    Merget_indices = connectivities[connectivities.connectivity.isin(Merget_connectivities)].index.tolist()
    Christmann_connectivities = sources[sources.source.str.contains('Christmann2016')].connectivity.unique()
    Christmann_indices = connectivities[connectivities.connectivity.isin(Christmann_connectivities)].index.tolist()
    # Free RAM
    del sources, connectivities, ChEMBL_connectivities, ExCAPE_connectivities, Sharma_connectivities
    del Klaeger_connectivities, Merget_connectivities, Christmann_connectivities
    # Construct UpSet plot
    data = from_contents(
        {'ExCAPE-DB': ExCAPE_indices, 'ChEMBL29': ChEMBL_indices, 'Merget et al. (2016)': Merget_indices,
         'Christmann-Franck et al. (2016)': Christmann_indices, 'Sharma et al. (2016)': Sharma_indices,
         'Klaeger et al. (2017)': Klaeger_indices})
    upset = UpSet(data, subset_size='count', sort_by='degree', min_subset_size=1, show_counts=True, logscale=True,
                  intersection_label_rotation=45)
    plot = upset.plot()
    # Save plot
    plt.savefig(plot_png, **kwargs)


def activity_upset_plot(sd_xz, tsv_xz, plot_png, **kwargs):
    # Read sources
    sources = pd.read_csv(tsv_xz, sep='\t', usecols=['source'])
    # Get rows with corresponding sources and identify their indices in the set of molecules
    ChEMBL_indices = sources[sources.source.str.contains('ChEMBL29')].index.tolist()
    ExCAPE_indices = sources[sources.source.str.contains('ExCAPE-DB')].index.tolist()
    Sharma_indices = sources[sources.source.str.contains('Sharma2016')].index.tolist()
    Klaeger_indices = sources[sources.source.str.contains('Klaeger2017')].index.tolist()
    Merget_indices = sources[sources.source.str.contains('Merget2017')].index.tolist()
    Christmann_indices = sources[sources.source.str.contains('Christmann2016')].index.tolist()
    # Free RAM
    del sources
    # Construct UpSet plot
    data = from_contents(
        {'ExCAPE-DB': ExCAPE_indices, 'ChEMBL29': ChEMBL_indices, 'Merget et al. (2016)': Merget_indices,
         'Christmann-Franck et al. (2016)': Christmann_indices, 'Sharma et al. (2016)': Sharma_indices,
         'Klaeger et al. (2017)': Klaeger_indices})
    upset = UpSet(data, subset_size='count', sort_by='degree', min_subset_size=1, show_counts=True, logscale=True,
                  intersection_label_rotation=45)
    plot = upset.plot()
    # Save plot
    plt.savefig(plot_png, **kwargs)


def target_upset_plot(activity_tsv_xz, target_tsv_xz, plot_png, **kwargs):
    # Read sources
    sources = pd.read_csv(activity_tsv_xz, sep='\t', usecols=['source', 'target_id'])
    # Get order of targets
    targets = pd.read_csv(target_tsv_xz, sep='\t', usecols=['target_id'])
    # Get rows with corresponding sources and identify their indices in the set of molecules
    ChEMBL_targets = sources[sources.source.str.contains('ChEMBL29')].target_id.unique()
    ChEMBL_indices = targets[targets.target_id.isin(ChEMBL_targets)].index.tolist()
    ExCAPE_targets = sources[sources.source.str.contains('ExCAPE-DB')].target_id.unique()
    ExCAPE_indices = targets[targets.target_id.isin(ExCAPE_targets)].index.tolist()
    Sharma_targets = sources[sources.source.str.contains('Sharma2016')].target_id.unique()
    Sharma_indices = targets[targets.target_id.isin(Sharma_targets)].index.tolist()
    Klaeger_targets = sources[sources.source.str.contains('Klaeger2017')].target_id.unique()
    Klaeger_indices = targets[targets.target_id.isin(Klaeger_targets)].index.tolist()
    Merget_targets = sources[sources.source.str.contains('Merget2017')].target_id.unique()
    Merget_indices = targets[targets.target_id.isin(Merget_targets)].index.tolist()
    Christmann_targets = sources[sources.source.str.contains('Christmann2016')].target_id.unique()
    Christmann_indices = targets[targets.target_id.isin(Christmann_targets)].index.tolist()
    # Free RAM
    del sources, targets, ChEMBL_targets, ExCAPE_targets, Sharma_targets
    del Klaeger_targets, Merget_targets, Christmann_targets
    # Construct UpSet plot
    data = from_contents(
        {'ExCAPE-DB': ExCAPE_indices, 'ChEMBL29': ChEMBL_indices, 'Merget et al. (2016)': Merget_indices,
         'Christmann-Franck et al. (2016)': Christmann_indices, 'Sharma et al. (2016)': Sharma_indices,
         'Klaeger et al. (2017)': Klaeger_indices})
    upset = UpSet(data, subset_size='count', sort_by='degree', min_subset_size=1, show_counts=True, logscale=True,
                  intersection_label_rotation=45)
    plot = upset.plot()
    # Save plot
    plt.savefig(plot_png, **kwargs)


def main(root_dir: str,
         out_dir: str,
         version: str):
    version = process_data_version(version)
    # Determine paths of input files
    if root_dir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_dir)
    papyrus_root = pystow.module('papyrus', version)
    sd_file = papyrus_root.join('structures', name=f'{version}_combined_2D_set_without_stereochemistry.sd.xz').as_posix()
    activity_file = papyrus_root.join(name=f'{version}_combined_set_without_stereochemistry.tsv.xz').as_posix()
    protein_file = papyrus_root.join(name=f'{version}_combined_set_protein_targets.tsv.xz').as_posix()
    # Create output directory
    outdir = os.path.join(os.path.abspath(out_dir), 'upset')
    os.makedirs(outdir, exist_ok=True)
    # Create upset plots
    compound_upset_plot(sd_file, activity_file,
                        os.path.join(outdir, 'Papyrus_upsetplot_molecularSpace.png'), dpi=1200)
    activity_upset_plot(sd_file, activity_file,
                        os.path.join(outdir, 'Papyrus_upsetplot_activitySpace.png'), dpi=1200)
    target_upset_plot(activity_file, protein_file,
                      os.path.join(outdir, 'Papyrus_upsetplot_targetSpace.png'), dpi=1200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate upset plots of the source dataset of Papyrus.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--indir',
                        required=False,
                        default=None,
                        help=('Directory containing Papyrus\'s data directory\n'
                              '(default: pystow\'s home folder).'),
                        dest='indir')
    parser.add_argument('-o', '--outdir',
                        default='./',
                        required=False,
                        help=('Directory where upset plots will be written\n'
                              '(default: upset in the current folder).'),
                        dest='outdir')
    parser.add_argument('-V', '--version',
                        default='latest',
                        required=False,
                        help=('Version of the Papyrus data to be used (default: latest).'),
                        dest='version')
    args = parser.parse_args()
    main(root_dir=args.indir,
         out_dir=args.outdir,
         version=args.version)
