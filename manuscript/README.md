
# Results
This directory contains commands to generate all figures presented in the associated preprint.

*It is considered that a conda environment has properly been setup and activated.*
____
 ### 1. Download the Papyrus data
Only the standardized data (i.e. without stereochemistry) is considered in the preprint (-s without).
Molecualr structures (-S) and all molecular descriptors are to be downloaded (-d all). 
```bash
python 00-Download.py -s without -S -d all
```
______
 ### 2. Obtain model performance
 
1. Model activities with standard QSAR and PCM models,
    
Create classification (-C), regression (-R) QSAR (-Q) and PCM (-P) models using each descriptor type one at a time splitting the data randomly (-s random) and temporally (-s year).
```bash
python 01-Modelling.py -C -R -Q -P -m mold2 cddd mordred fingerprint -s random year
```
    
2. Then, model activities with single-task DNN PCM models,

```bash
python 02-Modelling_DNNs_classification.py -m mold2 cddd mordred fingerprint -s random year
python 03-Modelling_DNNs_regression.py -m mold2 cddd mordred fingerprint -s random year
```
3. Finally, graph results.
```bash
python 04-Moddeling_plots.py -i results -d results_dnn -s random -e CV --summary CV_random.tsv avg_CV_random.tsv QSAR_CV_random.svg PCM_CV_random.svg DNN_CV_random.svg ALL_CV_random.svg
python 04-Moddeling_plots.py -i results -d results_dnn -s temporal -e CV --summary CV_temporal.tsv avg_CV_temporal.tsv QSAR_CV_temporal.svg PCM_CV_temporal.svg DNN_CV_temporal.svg ALL_CV_temporal.svg
python 04-Moddeling_plots.py -i results -d results_dnn -s random -e Test --summary test-set_random.tsv avg_test-set_random.tsv QSAR_test-set_random.svg PCM_test-set_random.svg DNN_test-set_random.svg ALL_test-set_random.svg
python 04-Moddeling_plots.py -i results -d results_dnn -s temporal -e Test --summary test-set_temporal.tsv avg_test-set_temporal.tsv QSAR_test-set_temporal.svg PCM_test-set_temporal.svg DNN_test-set_temporal.svg ALL_test-set_temporal.svg
```
______
 ### 3. Generate TMAPs of the chemical space
 Generate a TreeMap (TMAP) of the chemical space.
```bash
python 05-TMAP.py
```
______
### 4. Create phylogenetic trees of the target subsets

1. Obtain and compile [Kalign](https://github.com/TimoLassmann/kalign/releases/tag/v3.3.1), then create an alias,
```bash
current_dir=$PWD
wget https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.1.tar.gz
tar xfz v3.3.1.tar.gz && rm v3.3.1.tar.gz
cd kalign-3.3.1/ && ./autogen.sh && ./configure && make
export kalign=$PWD/src/kalign
cd $current_dir
```
2. Create the phylogenetic tree
It is recommended to run this script on a machine with 20+ cores.
```bash
python 06-Phylotree.py create -k "$kalign" --njobs 24
```
3. Use [FigTree](https://github.com/rambaut/figtree/releases/tag/v1.4.4) to generate a visualisation of the tree:
    - Once open, go to 'File > Open...' and select the *Human_protein_targets_Kalign_nj_tree.nwk*
    - Dismiss the message ahout labels of the node and branches
    - In the left tab
        - Under 'Layout', change the tree display to radial layout
        - Under 'Trees'
            - Root the tree using the midpoint
            - Transform the branches to 'cladogram'
        - Tick 'Tip Shapes'
        - Untick 'Scale Bar'
    - Export the visualisation to SVG with 'File > Export SVG...' into the same directory the *Human_protein_targets_Kalign_nj_tree.nwk* is in and name it *Kalign_nj_tree.svg*.
 
 One might be faced with the following error preventing from opening FigTree:
 ```Exception in thread "main" java.awt.AWTError: Can't connect to X11 window server...```
 Which can be addressed by changing the ```DISPLAY``` environment variable:
```bash
export DISPLAY=:0
```

5. Finally, post-process the generated SVG file to highlight the selected targets.
```bash
python 06-Phylotree.py process -f Kalign_nj_tree.svg -o ARs_Kalign_nj_tree.svg -H "{{'l5': 'Adenosine receptor'}: '#0072B2'}" --scaling 4
python 06-Phylotree.py process -f Kalign_nj_tree.svg -o CCRs_Kalign_nj_tree.svg -H "{{'l5': 'CC chemokine receptor'}: '#009E73'}" --scaling 4
python 06-Phylotree.py process -f Kalign_nj_tree.svg -o Kinase_Kalign_nj_tree.svg -H "{{'l2': 'Kinase'}: '#CC79A7'}" --scaling 4
python 06-Phylotree.py process -f Kalign_nj_tree.svg -o MRs_Kalign_nj_tree.svg -H "{{'l4': 'Monoamine receptor'}: '#F0E442'}" --scaling 4
python 06-Phylotree.py process -f Kalign_nj_tree.svg -o SCL6_Kalign_nj_tree.svg -H "{{'l4': 'SLC06 neurotransmitter transporter family'}: '#D55E00'}" --scaling 4
```
______
### 5. Generate upset plots of source datasets
```bash
python 07-Upset_plots.py
```
