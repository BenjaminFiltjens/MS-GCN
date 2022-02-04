# Multi-Stage Spatial-Temporal Convolutional Neural Network (MS-GCN)
This code implements the skeleton-based action segmentation MS-GCN model from [Automated freezing of gait assessment with
marker-based motion capture and multi-stage
spatial-temporal graph convolutional neural
networks](https://arxiv.org/abs/2103.15449) and [Skeleton-based action segmentation with multi-stage spatial-temporal graph convolutional neural networks](https://arxiv.org/abs/2202.01727), arXiv 2022 (in-review).

It was originally developed for freezing of gait (FOG) assessment on a [proprietary dataset](https://movementdisorders.onlinelibrary.wiley.com/doi/10.1002/mds.23327). Recently, we have also achieved high skeleton-based action segmentation performance on public datasets, e.g. [HuGaDB](https://arxiv.org/abs/1705.08506), [LARa](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7436169/), [PKU-MMD v2](https://arxiv.org/abs/1703.07475), [TUG](https://www.nature.com/articles/s41597-020-00627-7).

## Requirements
Tested on Ubuntu 16.04 and Pytorch 1.10.1. Models were trained on a
[Nvidia Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/).

The c3d data preparation script requires [Biomechanical-Toolkit](https://github.com/Biomechanical-ToolKit/BTKPython). For installation instructions, please refer to the following [issue](https://github.com/Biomechanical-ToolKit/BTKPython/issues/2).

## Content
* `data_prep/` -- Data preparation scripts.
* `main.py` -- Main script. I suggest working with this interactively with an IDE. Please provide the dataset and train/predict arguments, e.g. `--dataset=fog_example --action=train`.
* `batch_gen.py` -- Batch loader.
* `label_eval.py` -- Compute metrics and save prediction results.
* `model.py` -- train/predict script.
* `models/` -- Location for saving the trained models.
* `models/ms_gcn.py` -- The MS-GCN model.
* `models/net_utils/` -- Scripts to partition the graph for the various datasets. For more information about the partitioning, please refer to the section [Graph representations](). For more information about spatial-temporal graphs, please refer to [ST-GCN](https://arxiv.org/pdf/1801.07455.pdf).
* `data/` -- Location for the processed datasets. For more information, please refer to the 'FOG' example.
* `data/signals.` -- Scripts for computing the feature representations. Used for datasets that provided spatial features per joint, e.g. FOG, TUG, and PKU-MMD v2. For more information, please refer to the section [Graph representations]().
* `results/` -- Location for saving the results.

## Data
After processing the dataset (scripts are dataset specific), each processed dataset should be placed in the ``data`` folder. We provide an example for a motion capture dataset that is in [c3d](https://www.c3d.org/) format. For this particular example, we extract 9 joints in 3D:
* `data_prep/read_frame.py` -- Import the joints and action labels from the c3d and save both in a separate csv.
* `data_prep/gen_data/` -- Import the csv, construct the input, and save to npy for training. For more information about the input and label shape, please refer to the section [Problem statement]().

Please refer to the example in `data/example/` for more information on how to structure the files for training/prediction.

## Pre-trained models
Pre-trained models are provided for HuGaDB, PKU-MMD, and LARa. To reproduce the results from the paper:
* The dataset should be downloaded from their respective repository.
* See the "Data" section for more information on how to prepare the datasets.
* Place the pre-trained models in ``models/``, e.g. ``models/hugadb``.
* Ensure that the correct graph representation is chosen in ``ms_gcn``.
* Comment out ``features = get_features(features)`` in model (only for lara and hugadb).
* Specify the correct sampling rate, e.g. downsampling factor of 4 for lara.
* Run main to generate the per-sample predictions with proper arguments, e.g. ``--dataset=hugadb`` ``--action=predict``.
* Run label_eval with proper arguments, e.g. ``--dataset=hugadb``.

## Acknowledgements
The MS-GCN model and code are heavily based on [ST-GCN](https://github.com/yysijie/st-gcn) and [MS-TCN](https://github.com/yabufarha/ms-tcn). We thank the authors for publicly releasing their code.

## License
[MIT](https://choosealicense.com/licenses/mit/)
