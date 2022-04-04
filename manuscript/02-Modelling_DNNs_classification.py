# -*- coding: utf-8 -*-

import os
import argparse
from typing import List

import pystow
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from papyrus_scripts.modelling import (get_molecular_descriptors,
                                           filter_molecular_descriptors,
                                           get_protein_descriptors,
                                           model_metrics)
from papyrus_scripts.neuralnet import SingleTaskNNClassifier
from papyrus_scripts.utils.IO import process_data_version


# Handle required dependencies
try:
    import torch as T
except ImportError as e:
    raise ImportError('Some required dependencies are missing:\n\tpytorch') from e


# Handle pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'


def main(root_folder: str,
         subset_folder: str,
         out_folder: str,
         version: str,
         targets: List[str],
         moldescs: List[str],
         protdescs: List[str],
         splitby: List[str],
         seed: int,
         year: int,
         learning_rate: int,
         epochs: int,
         batchsize: int,
         earlystop: int,
         dropout: float,
         endpoint: str,
         unisize: int,
         gpu: List[int]):
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu)
    version = process_data_version(version)
    # Determine default paths
    if root_folder is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(root_folder)
    papyrus_root = pystow.module('papyrus', version)
    descriptor_folder = papyrus_root.join('descriptors')
    # Handle exceptions
    if not os.path.isdir(subset_folder):
        raise ValueError('directory of filtered subsets does not exist')
    subset_folder = os.path.join(os.path.abspath(subset_folder), 'results')
    if not os.path.isdir(subset_folder):
        raise ValueError('directory \'results\' in the subset folder does not exist')
    # Separate out folders of models and summarized results
    results_out_folder = os.path.join(os.path.abspath(out_folder), 'results_dnn')
    models_out_folder = f'{results_out_folder}/models/classification'
    os.makedirs(models_out_folder, exist_ok=True)
    # Identify targets
    datasets = []
    targets = [target.lower() for target in targets]
    if 'ars' in targets:
        datasets.append(('ARs', f'{subset_folder}/results_filtered_ARs_high.txt.xz', 1024, 300, 256))
    if 'ccrs' in targets:
        datasets.append(('CCRs', f'{subset_folder}/results_filtered_CCRs_human_high.txt.xz', 1024, 300, 256))
    if 'slc6' in targets:
        datasets.append(('SLC6', f'{subset_folder}/results_filtered_SLC6_high.txt.xz', 1024, 300, 256))
    if 'mrs' in targets:
        datasets.append(('MonoamineRs', f'{subset_folder}/results_filtered_MonoamineR_human_high.txt.xz', 64, 20, 64))
    if 'kinases' in targets:
        datasets.append(('Kinases', f'{subset_folder}/results_filtered_Kin_human_high.txt.xz', 30, 20, 64))
    # Iterate over parameters
    for target, dataset, batch_size, early_stop, uni_size in datasets:
        for mol_descriptor_type in moldescs:
            for prot_descriptor_type in protdescs:
                for split_by in splitby:
                    # Force given arguments
                    if earlystop is not None:
                        early_stop = earlystop
                    if batchsize is not None:
                        batch_size = batchsize
                    if unisize is not None:
                        uni_size = unisize
                    if mol_descriptor_type == 'fingerprint':
                        output = f'{results_out_folder}/DNNst_PCM_results_{target}_classification_ECFP_{prot_descriptor_type}_{split_by}.tsv'
                        dnn_out = f'{models_out_folder}/DNNst_PCM_{target}_classification_ECFP_{prot_descriptor_type}_{split_by}'
                    else:
                        output = f'{results_out_folder}/DNNst_PCM_results_{target}_classification_{mol_descriptor_type}_{prot_descriptor_type}_{split_by}.tsv'
                        dnn_out = f'{models_out_folder}/DNNst_PCM_{target}_classification_{mol_descriptor_type}_{prot_descriptor_type}_{split_by}'
                    print('\nWorking on DNN with the following params:')
                    print(tabulate([['target', target],
                                    ['model type', 'classification'],
                                    ['molecular descs', mol_descriptor_type],
                                    ['protein descs', f'{prot_descriptor_type}{uni_size}'],
                                    ['split by', split_by],
                                    ['year', year],
                                    ['epochs', epochs],
                                    ['learning rate', learning_rate],
                                    ['batch size', batch_size],
                                    ['early stop', early_stop],
                                    ['dropout', dropout],
                                    ['random seed', seed],
                                    ['gpu', os.environ['CUDA_VISIBLE_DEVICES']]]
                                   )
                          )
                    if os.path.isfile(output):
                        print('Results already obtained\n')
                        continue
                    print('Preparing dataset')
                    cls_dnn(dataset, endpoint, mol_descriptor_type, prot_descriptor_type,
                            descriptor_folder, split_by, uni_size, year, seed, output,
                            dnn_out, learning_rate, epochs, dropout, batch_size, early_stop)


def cls_dnn(dataset: str,
            endpoint: str,
            mol_descriptor_type: str,
            prot_descriptor_type: str,
            descriptor_folder:str,
            split_by: str,
            unisize: int,
            split_year: int,
            random_seed: int,
            perf_output: str,
            model_output: str,
            learning_rate: float,
            epochs: int,
            dropout: float,
            batch_size: int,
            early_stop: int):
    # Read dataset
    data = pd.read_csv(dataset, sep='\t')
    merge_on = 'connectivity' if 'connectivity' in data.columns else 'InChIKey'
    features_to_keep = [merge_on, 'target_id', endpoint, 'Year']
    # Keep binary values
    preserved = data[~data['Activity_class'].isna()]
    preserved = preserved.drop(
        columns=[col for col in preserved if col not in [merge_on, 'target_id', 'Activity_class', 'Year']])
    # Determine actives & inactives values from activity threshold
    active = data[
        data['Activity_class'].isna() & (data[endpoint] > 6.5) & ~data['relation'].str.contains(
            '<')][features_to_keep]
    active.loc[:, 'Activity_class'] = 'A'
    active.drop(columns=[endpoint], inplace=True)
    inactive = data[
        data['Activity_class'].isna() & (data[endpoint] <= 6.5) & ~data['relation'].str.contains(
            '>')][features_to_keep]
    inactive.loc[:, 'Activity_class'] = 'N'
    inactive.drop(columns=[endpoint], inplace=True)
    # Combine processed data
    data = pd.concat([preserved, active, inactive])
    # Change endpoint
    endpoint = 'Activity_class'
    del preserved, active, inactive
    # Obtain molecular descriptors and merge
    descs = get_molecular_descriptors('connectivity' not in data.columns,
                                      mol_descriptor_type,
                                      descriptor_folder, 100000)
    descs = filter_molecular_descriptors(descs, merge_on, data[merge_on].unique())
    data = data.merge(descs, on=merge_on)
    del descs
    data = data.drop(columns=[merge_on])
    # Obtain protein descriptors and merge
    prot_descs = get_protein_descriptors(prot_descriptor_type,
                                         descriptor_folder,
                                         ids=data['target_id'].unique())
    # Keep only relevant UniRep features
    if prot_descriptor_type == 'unirep':
        prot_descs = prot_descs.loc[:,
                     prot_descs.columns.str.contains(f'UniRep{unisize}') | (prot_descs.columns == 'target_id')]
    data = data.merge(prot_descs, on='target_id')
    del prot_descs
    data = data.drop(columns=['target_id'])
    if split_by == 'year':
        # Temporal split
        test_set = data[data['Year'] >= split_year]
        if test_set.empty:
            raise ValueError(f'no test data for temporal split at {split_year}')
        training_set = data[~data.index.isin(test_set.index)]
        training_set = training_set.drop(columns=['Year'])
        test_set = test_set.drop(columns=['Year'])
        # Split validation set
        training_set, validation_set = train_test_split(training_set, test_size=0.30,
                                                        random_state=random_seed)
    else:
        # Random split
        data = data.drop(columns=['Year'])
        training_set, test_set = train_test_split(data, test_size=0.30,
                                                  random_state=random_seed)
        training_set, validation_set = train_test_split(training_set, test_size=0.30,
                                                        random_state=random_seed)
    # Split training and test sets
    y_train, y_test, y_valid = training_set[endpoint], test_set[endpoint], validation_set[endpoint]
    X_train, X_test, X_valid = (training_set.drop(columns=endpoint),
                                test_set.drop(columns=endpoint),
                                validation_set.drop(columns=endpoint)
                                )
    del data
    # Scale data
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    X_test[X_test.columns] = scaler.fit_transform(X_test)
    X_valid[X_valid.columns] = scaler.fit_transform(X_valid)
    # Encode binary labels
    lblenc = LabelEncoder()
    y_train.loc[:] = lblenc.fit_transform(y_train)
    y_test.loc[:] = lblenc.transform(y_test)
    y_valid.loc[:] = lblenc.transform(y_valid)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_valid = y_valid.astype(np.int32)
    # Perform Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    pbar = tqdm(desc='Fitting model', total=kfold.n_splits + 1)
    performance = []
    for i, (train, test) in enumerate(kfold.split(X_train, y_train)):
        pbar.set_description(f'Fitting model on fold {i + 1}', refresh=True)
        # Create a fit the DNN
        nn = SingleTaskNNClassifier(f'{model_output}/fold{i + 1}', epochs=epochs,
                                    lr=learning_rate, early_stop=early_stop, batch_size=batch_size,
                                    dropout=dropout, random_seed=random_seed)
        nn.set_validation(X_valid, y_valid)
        nn.set_architecture(X_train.shape[1], 1)
        nn.fit(X_train.iloc[train, :], y_train.iloc[train])
        performance.append(model_metrics(nn, y_train.iloc[test], X_train.iloc[test, :]))
        pbar.update()
    # Organize result in a dataframe
    performance = pd.DataFrame(performance)
    performance.index = [f'Fold {i + 1}' for i in range(kfold.n_splits)]
    # Add average and sd of  performance
    performance.loc['Mean'] = [np.mean(performance[col]) if ':' not in col else '-' for col in performance]
    performance.loc['SD'] = [np.std(performance[col]) if ':' not in col else '-' for col in performance]
    # Fit model on the entire dataset
    pbar.set_description('Fitting model on entire training set', refresh=True)
    nn = SingleTaskNNClassifier(f'{model_output}/full', epochs=epochs, lr=learning_rate,
                                early_stop=early_stop, batch_size=batch_size,
                                dropout=dropout, random_seed=random_seed)
    nn.set_validation(X_valid, y_valid)
    nn.set_architecture(X_train.shape[1], 1)
    nn.fit(X_train, y_train)
    pbar.update()
    performance.loc['Test set'] = model_metrics(nn, y_test, X_test)
    performance.to_csv(perf_output, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model the Papyrus data with classification DNN PCM models.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--indir',
                        required=False,
                        default=None,
                        help=('Directory containing Papyrus\'s data directory\n'
                              '(default: pystow\'s home folder).'),
                        dest='indir')
    parser.add_argument('-f', '--filterdir',
                        default='./',
                        required=False,
                        help=('Directory where subsets have been written\n'
                              '(default: results folder in the current folder).'),
                        dest='filterdir')
    parser.add_argument('-o', '--outdir',
                        default='./',
                        required=False,
                        help=('Directory where modelling results will be stored\n'
                              '(default: results_dnn in the current folder).'),
                        dest='outdir')
    parser.add_argument('-V', '--version',
                        default='latest',
                        required=False,
                        help=('Version of the Papyrus data to be used (default: latest).'),
                        dest='version')
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
    parser.add_argument('--endpoint',
                        default='pchembl_value_Mean',
                        choices=['pchembl_value_Mean', 'pchembl_value_StdDev', 'pchembl_value_SEM',
                                 'pchembl_value_N', 'pchembl_value_Median', 'pchembl_value_MAD'],
                        required=False,
                        help='Endpoint to be modelled (default: pchembl_value_Mean).',
                        dest='endpoint')
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        required=False,
                        help='Seed for randomness (default: 1234).',
                        dest='seed')
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        required=False,
                        help='Number of epochs (default: 1000).',
                        dest='epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        required=False,
                        help='Learning rate (default: 1e-3).',
                        dest='learningrate')
    parser.add_argument('--batchsize',
                        type=int,
                        required=False,
                        help='Batch size\n'
                             '(default: 1024, but 24 for Monoamine receptors and 30 for Kinases).',
                        dest='batchsize')
    parser.add_argument('--earlystop',
                        type=int,
                        required=False,
                        help='Number of epochs after which to stop if converged\n'
                             '(default: 300, but 20 for Kinases and Monoamine receptors.',
                        dest='earlystop')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.25,
                        required=False,
                        help='Dropout (default: 0.25).',
                        dest='dropout')
    parser.add_argument('--unisize',
                        type=int,
                        choices=[64, 256, 1900],
                        default=None,
                        required=False,
                        help='Size of the UniRep latent space\n'
                             '(default: 256, but 64 for Kinases and Monoamine receptors).',
                        dest='unisize')
    parser.add_argument('--gpu',
                        type=int,
                        required=False,
                        nargs='+',
                        help='GPU IDs to be used.',
                        dest='gpu')
    args = parser.parse_args()
    main(root_folder=args.indir,
         subset_folder=args.filterdir,
         out_folder=args.outdir,
         version=args.version,
         targets=args.targets,
         moldescs=args.mdescs,
         protdescs=args.pdescs,
         splitby=args.splitby,
         seed=args.seed,
         year=args.year,
         learning_rate=args.learningrate,
         epochs=args.epochs,
         batchsize=args.batchsize,
         earlystop=args.earlystop,
         dropout=args.dropout,
         endpoint=args.endpoint,
         unisize=args.unisize,
         gpu=args.gpu)
