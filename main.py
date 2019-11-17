import torch
import numpy as np
from scripts.otherapproaches import run_algo,run_algo_permuted
from scripts.network import SimpleModel
from scripts.plot import plot
from scripts.process import get_formatted_dataset,get_all_sample,get_all_single_sample,put_data
#import xgboost as xgb

"""
def get_sample(sample=minus[0],sampletype="minus"):
def get_all_sample(sampletype="minus",which=0):
def get_all_single_sample(sampletype="minus",sample="CHU"):
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
minus = ["CHU", "AJL","bay", "qPJ", "KUH", "ale", "VIN", "ERE", "4ER"]
plus = ["syp", "JLb", "KBA", "zak", "ARS", "ABD", "BAS", "BAH", "MAM"]
#torch.set_default_tensor_type()
if __name__ == "__main__":
    #sample = get_all_single_sample(sampletype="plus",sample="syp")
    #keys = list(sample.keys())
    #first = np.array([sample[keys[3]].T])
    #first = torch.from_numpy(first).double().to(device)
    #out = model.forward(first)
    dataset = get_formatted_dataset()
    x = []
    y = []
    x_test = []
    y_test = []
    for i in range(0,14):        
        x.append(dataset[i]['data'].flatten())
        y.append(dataset[i]['label'])
    for i in range(14,len(dataset)):        
        x_test.append(dataset[i]['data'].flatten())
        y_test.append(dataset[i]['label']
        )    
    run_algo_permuted(x,y,x_test,y_test,algo="ada",num_iteration=3)
    #clf.predict(x_test)
    '''
    Plan of attack:
        Permuted Sampling tests:
            take 10 samples as train,
            8 left out as test.
    '''
    '''
    model = SimpleModel().double().to(device)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
'''