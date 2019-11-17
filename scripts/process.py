import os
import shutil
import pandas as pd
import numpy as np
minus = ["CHU", "AJL","bay", "qPJ", "KUH", "ale", "VIN", "ERE", "4ER"]
plus = ["syp", "JLb", "KBA", "zak", "ARS", "ABD", "BAS", "BAH", "MAM"]
def put_data():
    names = os.listdir("data")
    current_names = []
    for name in names:
        for m in minus:
            if m in name:
                try:
                    shutil.copyfile(os.path.join('data',name),os.path.join("minus",m,name))
                except FileNotFoundError:
                    os.mkdir(os.path.join('minus',m))
                    shutil.copyfile(os.path.join('data',name),os.path.join("minus",m,name))
        for p in plus:
            if p in name:
                try:
                    shutil.copyfile(os.path.join('data',name),os.path.join("plus",p,name))
                except FileNotFoundError:
                    os.mkdir(os.path.join('plus',p))
                    shutil.copyfile(os.path.join('data',name),os.path.join("plus",p,name))
def get_sample(sample=minus[0],sampletype="minus"):
    sample_path = os.path.join(sampletype,sample,os.listdir(os.path.join(sampletype,sample))[0])
    data = pd.read_csv(sample_path).values
    return data
def get_all_sample(sampletype="minus",which=0):
    all_data = {}
    for sample in os.listdir(sampletype):
        sample_path = os.path.join(sampletype,sample,os.listdir(os.path.join(sampletype,sample))[which])
        all_data[sample] = pd.read_csv(sample_path).values
    return all_data
def get_all_single_sample(sampletype="minus",sample="CHU"):
    all_data = {}
    for samplefile in os.listdir(os.path.join(sampletype,sample)):
        samplefile_path = os.path.join(sampletype,sample,samplefile)
        all_data[samplefile] = pd.read_csv(samplefile_path).values
    return all_data
def get_formatted_dataset():
    dataset = []
    for m in minus:
        concated = np.zeros((20000,20))
        single_all_data = get_all_single_sample("minus",m)
        for j in range(0,5):
            concated[j*4000:(j+1)*4000,:] = single_all_data[list(single_all_data.keys())[j]][:4000,1:21]
        dataset.append({"data":concated,
                        "label":0})    
    for p in plus:
        concated = np.zeros((20000,20))
        single_all_data = get_all_single_sample("plus",p)
        for j in range(0,5):
            concated[j*4000:(j+1)*4000,:] = single_all_data[list(single_all_data.keys())[j]][:4000,1:21]
        dataset.append({"data":concated,
                        "label":1})    
    return dataset
