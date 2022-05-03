# -*- coding: utf-8 -*-

import os
import glob
import argparse
from itertools import chain
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import LinearLocator, AutoMinorLocator


sns.set(font_scale=2.5, rc={'text.usetex' : True, 'grid.alpha': 0.2}, style="whitegrid")


def tabulate_results(pcm_qsar_folder: str, dnn_folder:str, evaluate: str = 'Test set') -> pd.DataFrame:
    """Gather performance metrics of QSAR, PCM and DNN models.

    :param pcm_qsar_folder: folder containing the results of the QSAR and PCM modelling
    :param dnn_folder: folder containing the results of the modelling using DNNs
    :param evaluate: part of the results to be summarized:
                       - 'Mean' for average cross-validation performance
                       - 'Test set' for performance on held-out test set
    :return: the tabulated results per descriptor and model type (QSAR results averaged per target class)
    """
    if not os.path.isdir(pcm_qsar_folder):
        raise ValueError(f'folder does not exist: {pcm_qsar_folder}')
    if not os.path.isdir(dnn_folder):
        raise ValueError(f'folder does not exist: {dnn_folder}')
    if not evaluate in ['Test set', 'Mean']:
        raise ValueError('evaluate value must be one of: [\'Test set\', \'Mean\']')
    results = pd.DataFrame(None,
                           columns=pd.MultiIndex.from_product([['random', 'year'],
                                                               ['ECFP_6', 'CDDD', 'Mold2', 'Mordred2D'],
                                                               ['MCC', 'r', 'RMSE']]
                                                              ),
                           index=pd.MultiIndex.from_product([['ARs', 'CCRs', 'Kinases', 'MRs', 'SLC6'],
                                                             ['QSAR', 'PCM', 'stDNN PCM']]
                                                            )
                           )
    for file_ in chain(glob.glob(os.path.join(pcm_qsar_folder, '*_results_*.tsv')),
                       glob.glob(os.path.join(dnn_folder, '*_results_*.tsv'))):
        # Identify the type of results
        if os.path.basename(file_).startswith('QSAR'):
            model = 'QSAR'
        elif os.path.basename(file_).startswith('PCM'):
            model = 'PCM'
        elif os.path.basename(file_).startswith('DNNst_PCM'):
            model = 'stDNN PCM'
        else:
            raise ValueError('model type (QSAR/PCM/stDNN PCM) could not be determined from file name')
        # Read the results
        if model == 'QSAR':
            data = pd.read_csv(file_, sep='\t', index_col=[0,1]).xs(evaluate, level=1)
        else:
            data = pd.read_csv(file_, sep='\t', index_col=[0]).xs(evaluate)
        # Identify the type of split
        if 'random' in file_:
            split = 'random'
        elif 'year' in file_:
            split = 'year'
        else:
            raise ValueError('data split type could not be determined from file name')
        # Identify the type of descriptor
        if 'ECFP' in file_:
            desc = 'ECFP_6'
        elif 'cddd' in file_:
            desc = 'CDDD'
        elif 'mold2' in file_:
            desc = 'Mold2'
        elif 'mordred' in file_:
            desc = 'Mordred2D'
        else:
            raise ValueError('molecular descriptor could not be determined from file name')
        # Identify the type of model
        if 'regression' in file_:
            metrics = [('r', 'Pearson r'), ('RMSE', 'RMSE')]
        elif 'classification' in file_:
            metrics = [('MCC', 'MCC')]
        else:
            raise ValueError('model type (regressor/classifier) could not be determined from file name')
        # Identify the target
        if 'ARs' in file_:
            target = 'ARs'
        elif 'CCRs' in file_:
            target = 'CCRs'
        elif 'Kinases' in file_:
            target = 'Kinases'
        elif 'MonoamineRs' in file_:
            target = 'MRs'
        elif 'SLC6' in file_:
            target = 'SLC6'
        else:
            raise ValueError('protein target could not be determined from file name')
        # Set the values in the right location
        for results_metric, file_metric in metrics:
            if model != 'QSAR':
                if isinstance(data.loc[file_metric], str):
                    results.loc[(target, model), (split, desc, results_metric)] = float(data.loc[file_metric].strip('[]'))
                else:
                    results.loc[(target, model), (split, desc, results_metric)] = data.loc[file_metric]
            else:
                results.loc[(target, model), (split, desc, results_metric)] = data.loc[:, file_metric].mean()
    return results


def average_over_descriptors(data: pd.DataFrame, split: str = 'random')\
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare modelling results for plotting.

    :param data: tabulated results
    :return: results averaged over descriptors for QSAR, PCM and DNN models
    """
    aggregated = pd.melt(data.reset_index(),
                         id_vars=['level_0', 'level_1']
                         ).rename(columns={'level_0': 'target class',
                                           'level_1': 'model',
                                           'variable_0': 'split',
                                           'variable_1': 'descriptor',
                                           'variable_2': 'variable'})
    QSAR = aggregated.loc[(aggregated.model == 'QSAR') & (aggregated.split == split)]
    PCM = aggregated.loc[(aggregated.model == 'PCM') & (aggregated.split == split)]
    DNN = aggregated.loc[(aggregated.model == 'stDNN PCM') & (aggregated.split == split)]
    complete = aggregated.loc[aggregated.split == split]
    return QSAR, PCM, DNN, complete


def plot_results(data: Union[pd.DataFrame, List[pd.DataFrame]], horizontal: bool = False) -> Figure:
    """Plot modelling results

    :param data: list of averaged tabulated results for each model type
    :param horizontal: arrange subplot horizontaly
    :return: matplotlib figure
    """
    # Cast input to list
    if not isinstance(data, list) and isinstance(data, pd.DataFrame):
        data = [data]
    elif not isinstance(data, list):
        raise TypeError('data must be a pandas dataframe or a list of dataframes')
    # Make scale of RMSE go from 0 to 1.5 when others go from 0 to 1.0
    scale = 2 / 3.0
    for df in data:
        mask = df.variable.isin(['RMSE'])
        df.loc[mask, 'value'] = df.loc[mask, 'value'] * scale
        df.loc[:, 'target class'] = df.loc[:, 'target class'].apply(lambda x: ' '.join(f'${y}$' for y in x.split()))
    # Plot data
    if len(data) == 1:
        if horizontal:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8.5))
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8.5))
        g = sns.barplot(x='target class', y='value', hue='variable',
                        errwidth=1, capsize=0.1, ci='sd',
                        data=data[0], ax=ax, palette=sns.color_palette("Set2"))
        ax.set_ylabel('$MCC$ $and$ $r_{Pearson}$')
        ax2 = ax.twinx()

        ax.set_ylim(0, 1.0)
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('$RMSE$')
        ax.set_xlabel('')

        ymajorLocator = LinearLocator(11)
        yminorLocator = AutoMinorLocator()
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        _ = ax2.set_yticks(ax.get_yticks())
        _ = ax2.set_yticklabels([f'${x:.2f}$' for x in ax.get_yticks() / scale])
        yminorLocator = LinearLocator(10 * 3 + 1)
        ax2.yaxis.set_minor_locator(yminorLocator)

        if horizontal:
            #_ = g.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
            sns.move_legend(g, 'center left', bbox_to_anchor=(1.15, 0.5), title=None)
            # LaTeX-ify the legend
            for t, l in zip(g.get_legend().texts, ['$MCC$', '$Pearson$ $r$', '$RMSE$']):
                t.set_text(l)
        else:
            sns.move_legend(g, 'upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, title=None, frameon=False)
            # LaTeX-ify the legend
            for t, l in zip(g.get_legend().texts, ['$MCC$', '$Pearson$ $r$', '$RMSE$']):
                t.set_text(l)

    else:
        # Switch from vertical to horizontal layout
        if horizontal:
            fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(len(data) * 14, 8.5))
        else:
            fig, axes = plt.subplots(nrows=len(data), ncols=1, figsize=(14, len(data) * 8.5))
        # Plot results in their own subplot
        for i, df in enumerate(data):
            ax1 = axes[i]
            g1 = sns.barplot(x='target class', y='value', hue='variable',
                             errwidth=1, capsize=0.1, ci='sd',
                             data=df, ax=ax1, palette=sns.color_palette("Set2"))
            ax1.set_ylabel('$MCC$ $and$ $r_{Pearson}$')
            ax2 = ax1.twinx()  # duplicate y axis

            ax1.set_ylim(0, 1.0)
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_ylabel('$RMSE$')
            ax1.set_xlabel('')

            ymajorLocator = LinearLocator(11)
            yminorLocator = AutoMinorLocator()
            ax1.yaxis.set_major_locator(ymajorLocator)
            ax1.yaxis.set_minor_locator(yminorLocator)
            _ = ax2.set_yticks(ax1.get_yticks())
            _ = ax2.set_yticklabels([f'${x:.2f}$' for x in ax1.get_yticks() / scale])
            yminorLocator = LinearLocator(10 * 3 + 1)
            ax2.yaxis.set_minor_locator(yminorLocator)

            # Hide legend unless last subplot
            if i < len(data) - 1:
                g1.legend([], [], frameon=False)
            elif horizontal:
                _ = g1.legend(loc='center left', bbox_to_anchor=(1.17, 0.5))
                # LaTeX-ify the legend
                for t, l in zip(g1.get_legend().texts, ['$MCC$', '$Pearson$ $r$', '$RMSE$']):
                    t.set_text(l)
            else:
                sns.move_legend(g1, 'upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, title=None, frameon=False)
                # LaTeX-ify the legend
                for t, l in zip(g1.get_legend().texts, ['$MCC$', '$Pearson$ $r$', '$RMSE$']):
                    t.set_text(l)
        plt.tight_layout()
    return fig


def main(pcm_qsar_folder: str,
         dnn_folder: str,
         evaluate: str,
         split: str,
         outputs: Tuple[str, str, str],
         horizontal: bool = False,
         summary: str = None):
    """Main function"""
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
        raise ValueError('four output file names must be provided')
    if summary is not None and (not isinstance(summary, (list, tuple)) or len(summary) != 2):
        raise ValueError('two summary file names must be provided')
    # Read in model performance summaries
    results = tabulate_results(pcm_qsar_folder=pcm_qsar_folder,
                               dnn_folder=dnn_folder,
                               evaluate=evaluate)
    # Average over descriptors
    QSAR, PCM, DNN, complete = average_over_descriptors(data=results, split=split)
    # Generate plots
    qsar_plot = plot_results(QSAR, horizontal=horizontal)
    pcm_plot = plot_results(PCM, horizontal=horizontal)
    dnn_plot = plot_results(DNN, horizontal=horizontal)
    all_plot = plot_results([QSAR, PCM, DNN], horizontal=horizontal)
    # Create output directory
    os.makedirs('results_plots', exist_ok=True)
    # Save plots
    qsar_plot.savefig(os.path.join('results_plots', outputs[0]))
    pcm_plot.savefig(os.path.join('results_plots', outputs[1]))
    dnn_plot.savefig(os.path.join('results_plots', outputs[2]))
    all_plot.savefig(os.path.join('results_plots', outputs[3]))
    # Save summary
    if summary is not None:
        # Write summary file (all descriptors)
        results.to_csv(os.path.join('results_plots', summary[0]), sep='\t')
        # Write summary averaged across descriptors
        complete.groupby(by=['target class', 'model', 'split', 'variable'])['value']\
            .aggregate('mean')\
            .reset_index()\
            .pivot(index=['target class', 'model'],
                   columns=['split', 'variable'],
                   values=['value'])\
            .to_csv(os.path.join('results_plots', summary[1]), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the Papyrus modelling results.',
                                     epilog=('Make sure the following dependencies are installed for LaTeX support: '
                                             'texlive-latex-extra texlive-fonts-recommended dvipng cm-super'))
    parser.add_argument('-i', '--indir',
                        required=False,
                        default=None,
                        help='Directory containing Papyrus\'s QSAR and PCM modelling results',
                        dest='indir')
    parser.add_argument('-d', '--dnn_indir',
                        required=False,
                        default=None,
                        help='Directory containing Papyrus\'s DNN modelling results',
                        dest='dnn_indir')
    parser.add_argument('OUT_FILES',
                        nargs=4,
                        help='Names of the 4 output SVG files (QSAR, PCM, DNN and altogehter respectively).')
    parser.add_argument('-e', '--eval',
                        default='Test set',
                        choices=['Test', 'CV'],
                        required=False,
                        help=('Results to be plotted (\'Test set\' for holdout set,'
                              ' \'Mean\' for average of cross-validation.'),
                        dest='eval')
    parser.add_argument('-s', '--split',
                        default='random',
                        choices=['random', 'temporal'],
                        required=False,
                        help=('Data split to be considered.'),
                        dest='split')
    parser.add_argument('--horizontal',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Make the combined plot be horizontal.')
    parser.add_argument('--summary',
                        default=None,
                        nargs=2,
                        required=False,
                        help=('Names of the tabulated summary and the averaged files\n'
                             '(default: not written to disk)'))
    args = parser.parse_args()
    main(pcm_qsar_folder=args.indir,
         dnn_folder=args.dnn_indir,
         evaluate='Test set' if args.eval == 'Test' else 'Mean',
         split= 'random' if args.split == 'random' else 'year',
         outputs=args.OUT_FILES,
         horizontal=args.horizontal,
         summary=args.summary)
