#In this file we will place the dataloaders and data access for this file
import torch
from torch import nn

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

#example:
#dataset = H5SeismicDataset(
#    filepath="/home/ctl051/228/Project/CustomProject/ECE228Project/data/Full.h5",
#    transform = transforms.Compose(
#        [SpecgramShaper(), SpecgramToTensor()]
#    )
#)
class H5SeismicDataset(Dataset):
    """Loads samples from H5 dataset for use in native PyTorch dataloader."""
    def __init__(self, filepath, transform=None):
        self.transform = transform
        self.filepath = filepath


    def __len__(self):
        with h5py.File(self.filepath, 'r') as f:
            DataSpec = '/4.0/Spectrogram'
            return f[DataSpec].len()


    def __getitem__(self, idx): 
        X = torch.from_numpy(self.read_h5(self.filepath, idx))
        if self.transform:
            X = self.transform(X)
        return idx, X


    def read_h5(self, filepath, idx):
        with h5py.File(filepath, 'r') as f:
            DataSpec = '/4.0/Spectrogram'
            return f[DataSpec][idx]
#takes in a Dataset object and returns a list of the [train,validation,test] samplers
def getSamplerTrainValidTestSplit(H5DatasetLen: int,valid_perc,test_perc) -> list:
    #Create a sampler for a random trian-test split
    
    indices = list(range(H5DatasetLen))
    valid_split = int(np.floor(valid_perc * H5DatasetLen))
    test_split = int(np.floor(test_perc * H5DatasetLen))
    #if shuffle: #In a function check to shuffle the data or not
    #np.random.seed(random_seed) #use a seed for reproducibility
    np.random.shuffle(indices)
    valid_idx, test_idx,train_idx = indices[:valid_split],indices[valid_split:valid_split+test_split], indices[valid_split+test_split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return [train_sampler,valid_sampler,test_sampler]

#batch size was 512, Heather changed to 1
def getDataloaderSplit(H5Dataset, valid_perc,test_perc, batch_size=512, num_workers=4, pin_memory=True):
    # If using CUDA or a GPU, num workers needs to be 1 for proper precetage split
    #Pin memory is true for correct random split on a CUDA machine.
    samplers = getSamplerTrainValidTestSplit(len(H5Dataset),valid_perc,test_perc)
    DEC_loader_train = DataLoader(H5Dataset, sampler=samplers[0], batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    DEC_loader_valid = DataLoader(H5Dataset, sampler=samplers[1], batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    DEC_loader_test =  DataLoader(H5Dataset, sampler=samplers[2], batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return [DEC_loader_train,DEC_loader_valid, DEC_loader_test]

class SpecgramShaper(object):
    """Crop & reshape data."""
    def __init__(self, n=None, o=None, transform='sample_norm_cent'):
        self.n = n
        self.o = o
        self.transform = transform


    def __call__(self, X):
        if self.n is not None and self.o is not None:
            N, O = X.shape
        else:
            X = X[:-1, 1:]
        if self.transform is not None:
            if self.transform == "sample_norm":
                X /= np.abs(X).max(axis=(0,1))
            elif self.transform == "sample_norm_cent":
                # X = (X - X.mean(axis=(0,1))) / \
                # np.abs(X).max(axis=(0,1))
                X = (X - X.mean()) / np.abs(X).max()
            else:
                raise ValueError("Unsupported transform.")
        else:
            print("Test failed.")

        X = np.expand_dims(X, axis=0)
        return X


class SpecgramToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, X):
        return torch.from_numpy(X)

    

    