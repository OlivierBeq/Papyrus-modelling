# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import xgboost

# Optional dependencies
try:
    import pystow
except ImportError as e:
    pystow = e

from papyrus_scripts.modelling import qsar, pcm
from papyrus_scripts.preprocess import keep_quality, keep_protein_class, keep_organism, consume_chunks
from papyrus_scripts.reader import read_papyrus, read_protein_set
from papyrus_scripts.utils.IO import process_data_version


if isinstance(pystow, ImportError):
    raise ImportError('\nSome required dependencies are missing:\n\tpystow')


def main(root_folder: Optional[str],
         out_folder: str,
         version: str,
         classification: bool,
         regression: bool,
         qsar: bool,
         pcm: bool,
         targets: List[str],
         moldescs: List[str],
         protdescs: List[str],
         splitby: List[str],
         seed: int,
         year: int,
         ignore_models: bool):
    if not os.path.isdir(out_folder):
        raise ValueError('out folder does not exist')
    version = process_data_version(version)
    out_folder = os.path.abspath(out_folder)
    # Determine default paths
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    papyrus_root = pystow.module('papyrus')
    root_folder = papyrus_root.base.as_posix()
    descriptor_folder = papyrus_root.join(version, 'descriptors').as_posix()
    # Create the subsets
    if len(glob.glob(os.path.join(out_folder, 'results_filtered*.txt.xz'))) != 5:
        create_datasets(root_folder, version, out_folder)
    # Set model types
    model_types = []
    if classification:
        model_types.append('classification')
    if regression:
        model_types.append('regression')
    if not classification and not regression:
        model_types.append('classification')
    # Identify targets
    datasets = []
    targets = [target.lower() for target in targets]
    if 'ars' in targets:
        datasets.append(('ARs', f'{out_folder}/results_filtered_ARs_high.txt.xz'))
    if 'ccrs' in targets:
        datasets.append(('CCRs', f'{out_folder}/results_filtered_CCRs_human_high.txt.xz'))
    if 'mrs' in targets:
        datasets.append(('MonoamineRs', f'{out_folder}/results_filtered_MonoamineR_human_high.txt.xz'))
    if 'slc6' in targets:
        datasets.append(('SLC6', f'{out_folder}/results_filtered_SLC6_high.txt.xz'))
    if 'kinases' in targets:
        datasets.append(('Kinases', f'{out_folder}/results_filtered_Kin_human_high.txt.xz'))
    # If QSAR is toggled or neither QSAR nor PCM
    if qsar or (not qsar and not pcm):
        build_qsar(descriptor_folder,
                   out_folder,
                   datasets,
                   model_types,
                   moldescs,
                   splitby,
                   seed,
                   year,
                   ignore_models)
    # If PCM is toggled
    if pcm:
        build_pcm(root_folder,
                  descriptor_folder,
                  out_folder,
                  datasets,
                  model_types,
                  moldescs,
                  protdescs,
                  splitby,
                  seed,
                  year,
                  ignore_models)


def create_datasets(root_folder, version, out_folder):
    # Read bioactivity data
    data = read_papyrus(is3d=False, version=version, chunksize=1000000, source_path=root_folder)
    # Read protein target information
    protein_data = read_protein_set(root_folder)
    # Adenosine Receptors:
    ## Apply data filters 
    filter1 = keep_quality(data, 'high')
    filter2 = keep_protein_class(filter1, protein_data, {'l5': 'Adenosine receptor'})
    data_filtered = consume_chunks(filter2, total=60)  # 60 = 59,763,781 aggregated points / 1,000,000 chunksize
    ## Save dataset to disk
    data_filtered.to_csv(f'{out_folder}/results_filtered_AR_high.txt.xz', sep='\t', index=False)
    del data_filtered
    # Kinases
    ## Apply data filters
    data = read_papyrus(is3d=False, version=version, chunksize=1000000, source_path=root_folder)
    filter1 = keep_quality(data, 'high')
    filter2 = keep_protein_class(filter1, protein_data, {'l2': 'Kinase'})
    filter3 = keep_organism(filter2, protein_data, 'Homo sapiens (Human)')
    data_filtered = consume_chunks(filter3, total=60)  # 60 = 59,763,781 aggregated points / 1,000,000 chunksize
    ## Save dataset to disk
    data_filtered.to_csv(f'{out_folder}/results_filtered_Kin_human_high.txt.xz', sep='\t', index=False)
    del data_filtered
    # SLC6 family
    ## Apply data filters
    data = read_papyrus(is3d=False, version=version, chunksize=1000000, source_path=root_folder)
    filter1 = keep_quality(data, 'high')
    filter2 = keep_protein_class(filter1, protein_data, {'l4': 'SLC06 neurotransmitter transporter family'})
    data_filtered = consume_chunks(filter2, total=60)  # 60 = 59,763,781 aggregated points / 1,000,000 chunksize
    ## Save dataset to disk
    data_filtered.to_csv(f'{out_folder}/results_filtered_SLC6_high.txt.xz', sep='\t', index=False)
    del data_filtered
    # CCRs
    ## Apply data filters
    data = read_papyrus(is3d=False, version=version, chunksize=1000000, source_path=root_folder)
    filter1 = keep_quality(data, 'high')
    filter2 = keep_protein_class(filter1, protein_data, {'l5': 'CC chemokine receptor'})
    filter3 = keep_organism(filter2, protein_data, 'Homo sapiens (Human)')
    data_filtered = consume_chunks(filter3, total=60)  # 60 = 59,763,781 aggregated points / 1,000,000 chunksize
    ## Save dataset to disk
    data_filtered.to_csv(f'{out_folder}/results_filtered_CCRs_human_high.txt.xz', sep='\t', index=False)
    del data_filtered
    # Monoamine receptors
    ## Apply data filters
    data = read_papyrus(is3d=False, version=version, chunksize=1000000, source_path=root_folder)
    filter1 = keep_quality(data, 'high')
    filter2 = keep_protein_class(filter1, protein_data, {'l4': 'Monoamine receptor'})
    filter3 = keep_organism(filter2, protein_data, 'Homo sapiens (Human)')
    data_filtered = consume_chunks(filter3, total=60)  # 60 = 59,763,781 aggregated points / 1,000,000 chunksize
    ## Save dataset to disk
    data_filtered.to_csv(f'{out_folder}/results_filtered_MonoamineR_human_high.txt.xz', sep='\t', index=False)
    del data_filtered


def build_qsar(descriptor_folder: str,
               out_folder: str,
               datasets: List[Tuple[str, str]],
               model_types: List[str],
               moldescs: List[str],
               splitby: List[str],
               seed: int,
               year: int,
               ignore_models: bool):
    """Build quantitative structure-activity relationship models.
    
    :param descriptor_folder: folder containing molecular descriptors
    :param out_folder: folder containing filtered data and where to write modelling results
    """
    for target, dataset in datasets:
        for model_type in model_types:
            for descriptor_type in moldescs:
                for split_by in splitby:
                    if descriptor_type == 'fingerprint':
                        output = f'{out_folder}/QSAR_results_{target}_{model_type}_ECFP_{split_by}.tsv'
                        modelname = f'{out_folder}/QSAR_model_{target}_{model_type}_ECFP_{split_by}.joblib.xz'
                    else:
                        output = f'{out_folder}/QSAR_results_{target}_{model_type}_{descriptor_type}_{split_by}.tsv'
                        modelname = f'{out_folder}/QSAR_model_{target}_{model_type}_{descriptor_type}_{split_by}.joblib.xz'
                    print(output)
                    if os.path.isfile(output):
                        continue
                    if model_type == 'regression':
                        model = xgboost.XGBRegressor(verbosity=0)
                        stratify = False
                    else:
                        model = xgboost.XGBClassifier(verbosity=0)
                        stratify = True
                    data = pd.read_csv(dataset, sep='\t')
                    cls_results, cls_models = qsar(data,
                                                   split_by=split_by,
                                                   split_year=year,
                                                   random_state=seed,
                                                   descriptors=descriptor_type,
                                                   descriptor_path=descriptor_folder,
                                                   verbose=True,
                                                   model=model,
                                                   stratify=stratify)
                    cls_results.to_csv(output, sep='\t')
                    if not ignore_models:
                        joblib.dump(cls_models, modelname, compress=('xz', 9), protocol=0)


def build_pcm(root_folder: str,
              descriptor_folder: str,
              out_folder: str,
              datasets: List[Tuple[str, str]],
              model_types: List[str],
              moldescs: List[str],
              protdescs: List[str],
              splitby: List[str],
              seed: int,
              year: int,
              ignore_models: bool):
    """Build proteo-chemometric models.
    
    :param root_folder: folder containing papyrus bioactivity data and protein target information
    :param descriptor_folder: folder containing molecular and protein descriptors
    :param out_folder: folder containing filtered data and where to write modelling results
    """
    for target, dataset in datasets:
        for model_type in model_types:
            for mol_descriptor_type in moldescs:
                for prot_descriptor_type in protdescs:
                    for split_by in splitby:
                        if mol_descriptor_type == 'fingerprint':
                            output = f'{out_folder}/PCM_results_{target}_{model_type}_ECFP_{prot_descriptor_type}_{split_by}.tsv'
                            modelname = f'{out_folder}/PCM_model_{target}_{model_type}_ECFP_{prot_descriptor_type}_{split_by}.joblib.xz'
                        else:
                            output = f'{out_folder}/PCM_results_{target}_{model_type}_{mol_descriptor_type}_{prot_descriptor_type}_{split_by}.tsv'
                            modelname = f'{out_folder}/PCM_model_{target}_{model_type}_{mol_descriptor_type}_{prot_descriptor_type}_{split_by}.joblib.xz'
                        print(output)
                        if os.path.isfile(output):
                            continue
                        if model_type == 'regression':
                            model = xgboost.XGBRegressor(verbosity=0)
                            stratify = False
                        else:
                            model = xgboost.XGBClassifier(verbosity=0)
                            stratify = True
                        data = pd.read_csv(dataset, sep='\t')
                        cls_results, cls_models = pcm(data,
                                                      split_by=split_by,
                                                      split_year=year,
                                                      random_state=seed,
                                                      mol_descriptors=mol_descriptor_type,
                                                      mol_descriptor_path=descriptor_folder,
                                                      prot_sequences_path=root_folder,
                                                      prot_descriptors=prot_descriptor_type,
                                                      prot_descriptor_path=descriptor_folder,
                                                      verbose=True,
                                                      model=model,
                                                      stratify=stratify)
                        cls_results.to_csv(output, sep='\t')
                        if not ignore_models:
                            joblib.dump(cls_models, modelname, compress=('xz', 9), protocol=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model the Papyrus data with QSAR and PCM models.',
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
                        help=('Directory where modelling results will be stored\n'
                              '(default: current folder).'),
                        dest='outdir')
    parser.add_argument('-V', '--version',
                        default='latest',
                        required=False,
                        help=('Version of the Papyrus data to be used (default: latest).'),
                        dest='version')
    parser.add_argument('-C',
                        default=False,
                        action='store_true',
                        help=('Toggle creation of classification models\n'
                              '(if neither QSAR nor PCM toggled, then defaults to QSAR).'),
                        dest='classif')
    parser.add_argument('-R',
                        default=False,
                        action='store_true',
                        help='Toggle creation of regression models',
                        dest='regres')
    parser.add_argument('-Q',
                        default=False,
                        action='store_true',
                        help='Toggle creation of QSAR models',
                        dest='qsar')
    parser.add_argument('-P',
                        default=False,
                        action='store_true',
                        help='Toggle creation of PCM models',
                        dest='pcm')
    parser.add_argument('-m',
                        choices=['mold2', 'cddd', 'mordred', 'fingerprint', 'all'],
                        default=['fingerprint'],
                        nargs='+',
                        required=False,
                        help=('Type of molecular descriptors to be used for modelling:\n'
                              '    - mold2: 2D Mold2 molecular descriptors (777),\n'
                              '    - cddd: 2D continuous data-driven descriptors (512),\n'
                              '    - mordred: 2D mordred molecular descriptors (1613),\n'
                              '    - fingerprint: 2D RDKit Morgan fingerprint with radius 3\n'
                              '                   and 2048 bits,\n'
                              '    - all: all descriptors.\n'),
                        dest='mdescs')
    parser.add_argument('-p',
                        choices=['unirep'],
                        default=['unirep'],
                        nargs='+',
                        required=False,
                        help=('Type of protein descriptors to be used for modelling:\n'
                              '    - unirep: deep-learning protein sequence descriptors (6660).\n'),
                        dest='pdescs')
    parser.add_argument('-s', '--splitby',
                        choices=['random', 'year'],
                        default=['random'],
                        nargs='+',
                        required=False,
                        help=('Type of split for modelling:\n'
                              '    - random: randomly split the data into a training\n'
                              '              set (for CV) and a hold-out test set,\n'
                              '    - year: split the training set (for CV) and\n'
                              '            hold-out test set using a temporal split.\n'),
                        dest='splitby')
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        required=False,
                        help='Seed for randomness (default: 1234).',
                        dest='seed')
    parser.add_argument('--year',
                        type=int,
                        default=2013,
                        required=False,
                        help='Year for temporal split (default: 2013).',
                        dest='year')
    parser.add_argument('-t', '--target',
                        choices=['ARs', 'CCRs', 'MRs', 'SLC6', 'Kinases'],
                        default=['ARs', 'CCRs', 'MRs', 'SLC6'],
                        nargs='+',
                        required=False,
                        help=('Subsets to be modelled:\n'
                              '    - ARs: adenosine receptor data of high quality,\n'
                              '    - CCRs: human C-C chemokine receptor data of high quality,\n'
                              '    - MRs: human monoamine receptor data of high quality,\n'
                              '    - SLC6: solute carrier 6 transport family data of high quality,\n'
                              '    - Kinases: human kinase data of high quality.\n'),
                        dest='targets')
    parser.add_argument('--nosave',
                        default=False,
                        action='store_true',
                        help='Should models not be saved to disk.',
                        dest='nosave')
    args = parser.parse_args()
    main(root_folder=args.indir,
         out_folder=args.outdir,
         version=args.version,
         classification=args.classif,
         regression=args.regres,
         qsar=args.qsar,
         pcm=args.pcm,
         targets=args.targets,
         moldescs=args.mdescs,
         protdescs=args.pdescs,
         splitby=args.splitby,
         seed=args.seed,
         year=args.year,
         ignore_models=args.nosave)
