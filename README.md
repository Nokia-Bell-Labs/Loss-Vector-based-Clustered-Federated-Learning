# CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings 

This repository contains the official code of the paper ``CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings``, accepted at the Forty-Third International Conference on Machine Learning (ICML 2026).

We release our implementation of the following algorithms:
* CLoVE
* IFCA
* FedAvg
* Local-only
* Centralized

### Requirements

This code has been tested on Python 3.12.9.

Install the required packages:
```
pip install -r requirements.txt
```

Clone the FINCH repository:
```
git clone https://github.com/ssarfraz/FINCH-Clustering.git
```
This will create a `FINCH-Clustering` directory.
Copy the `__init__.py` and `finch.py` from CLoVE's `FINCH-patch` directory to `FINCH-Clustering/finch/` and overwrite the existing two files with the same name.

Install FINCH package

```
cd FINCH-Clustering
pip install .
```

This completes the package installs. You can now delete the `FINCH-Clustering` if you want.

The code should run on both CPU and GPU servers.
You may need to set an environment variable on a GPU server:

```CUBLAS_WORKSPACE_CONFIG=:4096:8 python mlflow_driver.py args```

If you have problems running it on a GPU server, then use a CPU server.

### Datasets

The code uses different external publicly available datasets:
* MNIST
* CIFAR-10
* Fashion-MNIST (FMNIST)
* Amazon Review
* AG News
* FEMNIST

All datasets are downloaded automatically when running the code, except for FEMNIST.
FEMNIST has to be downloaded before running the code from https://github.com/TalwalkarLab/leaf.git. Then, the following preprocessing command has to be run in order to get the samples we base our experiments on:
```
  cd leaf/data/femnist
  ./preprocess.sh -s niid --sf 0.3 -k 0 -t sample -tf 0.9 -smplseed 1549786595 -spltseed 1549786796
 ```

### Reproducing the experiments

All of the experiment configurations with all varying parameters we run are included in the folder `experiment_driver` inside the CSV file `experiment_list.csv`. 
Each experiment has a unique experiment ID.
The parameters that are not contained in the CSV file are set to their default values, which can be found in the `config.py` file.
Moreover, some of these default parameters in `config.py` are overridden for some of the experiments/datasets by the file `config_driver.py`.

To reproduce our results for an individual experiment with experiment ID `exp_id`, use this command:
```
python mlflow_driver.py --id exp_id
```

To run your own experiment, create a new row in the CSV file with your desired configuration and a numeric exp_id, save it, and run the above command with your defined `exp_id`. 

Results and logs will be stored in the `results` folder, in a subfolder with a numeric ID that corresponds to the experiment's respective row from the `experiment_list.csv` file.

For the supervised experiments, column "Exp Selector" indicates the dataset partitioning variations, as described in Section 5.1 and Appendix E in the paper. In particular, for the three image-based datasets (MNIST, CIFAR-10 and FMNIST), the correspondence is as follows:
* Label skew 1 &rarr; exp203
  * _No overlap, 2 classes per cluster_: clients within each cluster receive data from only two unique classes, with no label overlap between clients in different clusters.
* Label skew 2 &rarr; exp201
  * _Moderate Overlap, 4 classes per cluster, 2 classes overlapping_: each client within each cluster receives samples from 2 classes, with at least 1 label shared between the labels assigned to any two clusters.

* Label skew 3 &rarr; exp202 
  * _High Overlap, 9-10 classes per cluster_: there are 5 clusters with 5 clients each. Among the first 4 clusters, all clients of a cluster get samples from all 10 classes except one. The missing class is uniquely selected for each of the 4 clusters. The clients of the 5th cluster get samples from all 10 classes. The data of a class is distributed amongst the clients that are assigned to that class by sampling from a  Dirichlet distribution with parameter $\alpha=0.5$.
* Label skew 4 &rarr; exp501
  * _Dominant class with overlap_: 1/3 of data for clients belonging to a cluster is uniformly sampled from all classes, and the remaining 2/3 comes from a dominant class that is unique for each cluster. The data of a class is distributed amongst the clients that are assigned to that class by sampling from a  Dirichlet distribution with parameter $\alpha=0.5$.
* Feature skew &rarr; exp301
  * We apply image rotations (0°, 90°, 180°, 270°) to the MNIST dataset. Each client within a cluster is assigned data with a specific rotation. 
* Concept shift &rarr; exp111
  * Concept shift is achieved through label permutation: all labels are distributed across all clusters, but clients within each cluster receive data with two unique label swaps.


For the text-based datasets, the correspondence is as follows:
* Experiments with AmazonReview &rarr; expAmazon 
  * _no mixture, no overlap_
* Experiments with AG News &rarr; exp401
  * _overlap, 2 classes per cluster, 1 overlap between each cluster_: each client within each cluster receives samples from 4 classes, with at least 2 labels shared between the labels assigned to any two clusters.

For the ablation studies (runs 1900-1903, exp602), the column Ablation_Param is as follows: 0 indicates the default configuration, 1 the "no matching" experiment, 2 the "agglomerative clustering" experiment, and 3 the "square root loss" experiment, as described in Appendix E.5 of the paper.

For the unknown-K experiment with FEMNIST (expFEMNIST – Appendix E.1.4), there is no mixing: each human writer corresponds to one client.

For the partial participation experiments (exp701 – Appendix E.1.6), in terms of mixing, we followed the same process as in Label skew 1 (exp203). 

Additional details on the configuration used for these experiments can be found in the dictionary ```dict_overlap_swap_exps``` in the file ```utils/dataset_modifier.py```.

### Reproducing the tables

To reproduce the paper's tables based on the results:
For the supervised results, run the notebook ```exp_evaluations_supervised.ipynb```.
For the unsupervised results, run the notebook ```exp_evaluations_unsupervised.ipynb```.
Both are located in the ```plot_utils``` folder. 
The last few cells of both files allow the user to select which run IDs or which experiment selectors from the CSV to run the processing for.
The processed results in table format are stored as CSV files in the ```results``` folder or an ```extensions``` subfolder with a descriptive name that includes the exp_selector and the table's metric (e.g. "supervised_table_exp301_accuracies").
  


### Reproducing the figures

To reproduce the paper's figures based on the results:

* For Figure 3b (k-FED), run the Jupyter notebook `kfed_experiment.ipynb` in folder `plot_utils`.
* For all other figures, run the Jupyter notebook `plots_creator.ipynb` in folder `plot_utils`.

Figures are stored in a `media` folder.


### Citation

If you find this repository useful, please cite our paper:

```
@inproceedings{bhatia-papadis-CLoVE-2026,
  title={CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings},
  author={Randeep Bhatia and Nikos Papadis and Murali Kodialam and TV Lakshman and Sayak Chakrabarty},
  booktitle={Forty-Third International Conference on Machine Learning},
  year={2026},
}
```
