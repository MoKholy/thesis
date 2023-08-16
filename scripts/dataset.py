import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import torch

DATASETS_FOLDER_PATH = "./data/datasets/"

# ensure that the datasets folder exists
if not os.path.isdir(DATASETS_FOLDER_PATH):
    os.mkdir(DATASETS_FOLDER_PATH)
if not os.path.isdir(DATASETS_FOLDER_PATH + "processed"):
    os.mkdir(DATASETS_FOLDER_PATH + "processed")
if not os.path.isdir(DATASETS_FOLDER_PATH + "raw"):
    os.mkdir(DATASETS_FOLDER_PATH + "raw")



class CustomDataset(Dataset):

    def __init__(self, data, transform=None):
        self.features = data.iloc[:, :-2]
        self.labels = data.iloc[:, -1]
        self.string_labels = data.iloc[:, -2]
        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset.
        Returns:
            int: Length of the dataset.
        """
        return len(self.features)

    def get_string_label(self, idx):
        """Returns the string label at the given index.
        Args:
            idx (int): Index of the item to return.
        Returns:
            (str): String label at the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get item at index
        return self.string_labels.iloc[idx].tolist()
    
    def __getitem__(self, idx):
        """Returns the item at the given index.
        Args:
            idx (int): Index of the item to return.
        Returns:
            (torch.Tensor, torch.Tensor): Item at the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get item at index
        sample = self.features.iloc[idx, :]
        label = self.labels.iloc[idx]
        # apply transform if given
        if self.transform:
            sample = self.transform(sample)

        # change sample and label to tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        return sample, label


def split_dataset(data, test_ratio=0.1, val_ratio=0.1, stratify=True, random_state=42):
    """Split a dataset into train, validation and test set.
    Args:
        data (Dataframe): Dataset to split.
        stratify (bool): Whether to use stratified sampling.
    
    Returns:
        (Dataframe, Dataframe, Dataframe): Train, validation and test dataset.
    """
    # split data into train and test
    train, test = train_test_split(data, test_size=test_ratio, random_state=random_state, stratify=data.iloc[:, -1]) if stratify else train_test_split(data, test_size=test_ratio, random_state=random_state)
    

    # calculate new val ratio
    val_ratio = val_ratio / (1 - test_ratio)
    # split train into train and val
    train, val = train_test_split(train, test_size=val_ratio, random_state=random_state, stratify=train.iloc[:, -1]) if stratify else train_test_split(train, test_size=0.2, random_state=random_state)
    
    return train, val, test

# returns train, val and test data as a torch dataset
def load_dataset(name, stratify=True, random_state=42, test_ratio=0.1, val_ratio=0.1):
    """Load a dataset from the datasets folder.
    Args:
        name (str): Name of the dataset to load.
        
    Returns:
        (Dataset, Dataset, Dataset): Train, validation and test dataset.
    """

    # check if dataset exists in processed subfolder
    if os.path.isfile(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_train.parquet")) and \
        os.path.isfile(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_val.parquet")) and \
        os.path.isfile(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_test.parquet")):
        # load dataset
        train = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_train.parquet"))
        val = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_val.parquet"))
        test = pd.read_parquet(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_test.parquet"))

        print(f"Loaded dataset {name} from processed folder.")
        # return datasets
        return CustomDataset(train), CustomDataset(val), CustomDataset(test)
    
    # create dataset path using name
    dataset_path = os.path.join(DATASETS_FOLDER_PATH, "raw", name)

    
    # check if dataset file is parquet or csv
    if os.path.isfile(dataset_path + ".parquet"):
        # load dataset
        data = pd.read_parquet(dataset_path + ".parquet")
    elif os.path.isfile(dataset_path + ".csv"):
        # load dataset
        data = pd.read_csv(dataset_path + ".csv")
    else:
        raise Exception("Dataset not found.")
    
    # split data into train, val and test
    train, val, test = split_dataset(data, test_ratio=test_ratio, val_ratio=val_ratio, stratify=stratify, random_state=random_state)

    # save datasets
    train.to_parquet(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_train.parquet"))
    val.to_parquet(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_val.parquet"))
    test.to_parquet(os.path.join(DATASETS_FOLDER_PATH, "processed", name + "_test.parquet"))

    # return datasets
    return CustomDataset(train), CustomDataset(val), CustomDataset(test)


def test():
    # load final_dataset
    train, val, test = load_dataset("correctly_labeled", stratify=True, random_state=42, test_ratio=0.1, val_ratio=0.1)
    
    # print dataset sizes
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")

    print(f"Train vector shape {train[0][0].shape[0]}")



if __name__ == "__main__":
    test()