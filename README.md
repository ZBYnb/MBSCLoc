# MBSCLoc
Code of paper: MBSCLoc: Multi-label subcellular localization prediction based on cluster balanced subspace partitioning method and multi-class contrastive representation learning

You can use MBSLoc directly by accessing:http://www.mbscloc.com/model_predict

## File Descriptions：
feature_ex_gpu.py: This script enables GPU-accelerated encoding of features. It is recommended to use a GPU with at least 16GB of VRAM for acceleration. In this study, a GPU with 24GB of VRAM (such as the NVIDIA 4090) was used. Due to the presence of some randomness in the parameters of the large model, please ensure to set a random seed. The random seed for this study is set to 42.  

label_cl.py: This file handles label preprocessing. It loads the corresponding labels for the data and saves them as .npy files.   

model_saved_max_upsampling_9: Contains models using the maximum upsampling method with a subspace count of 9. This directory includes nine models.   

model_saved_min_downsampling_13: Contains models using the minimum downsampling method with a subspace count of 13. This directory includes thirteen models.   

test_npy: This folder contains the test data and test labels.   

mod: Includes the UTR-LM pre-trained model downloaded from HuggingFace, which is used for encoding mRNA.   

MFL.ipynb: The code in this Jupyter Notebook can be used to load models from either model_saved_max_upsampling_9 or model_saved_min_downsampling_13 for predicting the subcellular localization sites of mRNA.   

The data file is too large, please email me if you need all the raw data.

## Contact  
zhangbangyi@stu.jiangnan.edu.cn  
zby_9826@163.com
