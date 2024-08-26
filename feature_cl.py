import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os

from multimolecule import RnaTokenizer, UtrLmModel
tokenizer = RnaTokenizer.from_pretrained(r'mod')
fea_extract = UtrLmModel.from_pretrained(r'mod')


def label_to_array(label):
    return [int(char) for char in label]

def feature_pre(file_path,output_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].str.replace('>', '', 1)
    labels = df['label'].apply(lambda x: x[1:] if x.startswith(">") else x).tolist()
    key = str(labels[0])
    texts = df['data'].tolist()
    features = []
    for text in tqdm(texts):
        input = tokenizer(text, return_tensors='pt')
        output = fea_extract(**input)
        features.append(output.pooler_output.detach().numpy())
    features = np.array(features)
    features = features.reshape(features.shape[0], -1)
    output_name = 'mrna_features'+key+'.npy'
    path = os.path.join(output_path,output_name)
    print(f'SAVED TO {path}')
    np.save(path, features)

folder_path = r'Folder_Name'
for filename in tqdm(os.listdir(folder_path)):
    if os.path.isfile(os.path.join(folder_path, filename)):
        file_path = os.path.join(folder_path, filename)
        output_path = r'feature'
        result = feature_pre(file_path, output_path)


