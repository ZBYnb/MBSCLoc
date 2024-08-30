import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
import numpy as np
from multimolecule import RnaTokenizer, UtrLmModel

# Set random seeds to ensure consistent results
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call function to set random seed
set_seed()

# Load the word divider and model
tokenizer = RnaTokenizer.from_pretrained(r'mod')
fea_extract = UtrLmModel.from_pretrained(r'mod').cuda()

# Defines a function that converts a label string to an array of integers
def label_to_array(label):
    return [int(char) for char in label]

def feature_pre(file_path, output_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].str.replace('>', '', 1)
    labels = df['label'].apply(lambda x: x[1:] if x.startswith(">") else x).tolist()
    key = str(labels[0])
    texts = df['data'].tolist()

    # Extract feature
    features = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = fea_extract(**inputs)
        features.append(outputs.pooler_output.cpu().numpy())
    features = np.array(features)
    features = features.reshape(features.shape[0], -1)
    output_name = 'mrna_features' + key + '.npy'
    path = os.path.join(output_path, output_name)
    print(f'Saved to {path}')
    np.save(path, features)


# Specify folder path
folder_path = r'data'
# Go through all the files in the folder
for filename in tqdm(os.listdir(folder_path)):
    if os.path.isfile(os.path.join(folder_path, filename)):
        file_path = os.path.join(folder_path, filename)
        output_path = r'D:\zhangbangyi\code\UTR-LM-EX_web\traindata_feature'
        feature_pre(file_path, output_path)



# Specifies the path to the folder where the.npy file will be stored
input_folder = r'traindata_feature'

# Prepare a list of all features to store
all_features = []

# Gets all the.npy files in the folder
npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

# Walk through each.npy file, loading and adding them to the all_features list
for npy_file in tqdm(npy_files, desc="Loading npy files"):
    print(npy_file)
    file_path = os.path.join(input_folder, npy_file)
    features = np.load(file_path)
    all_features.append(features)

# Stack all features vertically into one large NumPy array
all_features = np.vstack(all_features)

# Save the merged features to a.npy file
output_path = r'features_train.npy'
print(f'Saved to {output_path}')
np.save(output_path, all_features)

