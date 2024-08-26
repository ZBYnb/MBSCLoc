import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os

def label_to_array(label):
    return [int(char) for char in label]

def label_pre(file_path,output_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].str.replace('>', '', 1)
    labels = df['label'].apply(lambda x: x[1:] if x.startswith(">") else x).tolist()
    key = str(labels[0])
    labels = [label_to_array(label) for label in labels]
    print(f'label_processing:{labels[0]}')
    labels = np.array(labels)
    output_name = 'labels_test'+key+'.npy'
    path = os.path.join(output_path,output_name)
    print(f'SAVED TO {path}')
    np.save(path, labels)



folder_path = r'Folder_Name'
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        file_path = os.path.join(folder_path, filename)
        output_path = r'labels'
        result = label_pre(file_path, output_path)
        print(result)

# test = np.load(r'labels_test/labels_test110010000.npy')

