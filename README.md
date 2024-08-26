# MBSCLoc
Code of paper: MBSCLoc: Multi-label subcellular localization prediction based on cluster balanced subspace partitioning method and multi-class contrastive representation learning

File Descriptions
feature_cl.py: This file is responsible for preprocessing data. It loads UTR-LM encoded mRNA sequence data into features and saves them as .npy files.

feature_ex_gpu.py: This script enables GPU-accelerated encoding of features. It is recommended to use a GPU with at least 16GB of VRAM for acceleration. In this study, a GPU with 24GB of VRAM (such as the NVIDIA 4090) was used.

label_cl.py: This file handles label preprocessing. It loads the corresponding labels for the data and saves them as .npy files.

model_saved_max_upsampling_9: Contains models using the maximum upsampling method with a subspace count of 9. This directory includes nine models.

model_saved_min_downsampling_13: Contains models using the minimum downsampling method with a subspace count of 13. This directory includes thirteen models.

test_npy: This folder contains the test data and test labels.

mod: Includes the UTR-LM pre-trained model downloaded from HuggingFace, which is used for encoding mRNA.

MFL.ipynb: The code in this Jupyter Notebook can be used to load models from either model_saved_max_upsampling_9 or model_saved_min_downsampling_13 for predicting the subcellular localization sites of mRNA.
