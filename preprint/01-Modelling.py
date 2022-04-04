import os.path

import pandas as pd
import xgboost

from src.papyrus_scripts.modelling import qsar, pcm
from src.papyrus_scripts.preprocess import keep_quality, keep_protein_class, consume_chunks
from src.papyrus_scripts.reader import read_papyrus, read_protein_set

if __name__ == '__main__':
    # Change path to folder where Papyrus data was donwloaded
    folder = 'F:/Downloads/Papyrus 05.4/'

    # Read bioactivity data
    data = read_papyrus(is3d=False, chunksize=50000, source_path=folder)
    # Read protein target information
    protein_data = read_protein_set(folder)

    # Apply data filters
    filter1 = keep_quality(data, 'high')
    filter4 = keep_protein_class(filter1, protein_data, {'l5': 'Adenosine receptor'})
    data_filtered = consume_chunks(filter4, total=1196)  # 1196 = 59,763,781 aggregated points / 50,000 chunksize

    # Save dataset to disk
    data_filtered.to_csv('results_filtered_adenosines_high.txt', sep='\t')

    # #QSAR modelling
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results_filtered_adenosines_high.txt'), sep='\t',
                       index_col=0).sort_values('target_id')
    cls_results, cls_models = qsar(data, descriptor_path=folder, verbose=True,
                                   model=xgboost.XGBClassifier(verbosity=0), stratify=True)
    cls_results.to_csv('QSAR_results_classification.txt', sep='\t')
    reg_results, reg_models = qsar(data, descriptor_path=folder, verbose=True,
                                   model=xgboost.XGBRegressor(verbosity=0))
    reg_results.to_csv('QSAR_results_regression.txt', sep='\t')

    # PCM modelling
    pcm_cls_result, pcm_cls_model = pcm(data, mol_descriptor_path=folder, prot_descriptor_path=folder,
                                        verbose=True, model=xgboost.XGBClassifier(verbosity=0), stratify=True)
    pcm_cls_result.to_csv('PCM_results_classification.txt', sep='\t')
    pcm_reg_result, pcm_reg_model = pcm(data, mol_descriptor_path=folder, prot_descriptor_path=folder,
                                        verbose=True, model=xgboost.XGBRegressor(verbosity=0))
    pcm_reg_result.to_csv('PCM_results_regression.txt', sep='\t')
