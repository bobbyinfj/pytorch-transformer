""" The loader_customtext.py file loads in files from the Flicker8k Caption Dataset using the PyTorch DataLoader class and performs the following steps to ready the data for use in machine learning models.

To convert text -> numerical values
1. Need a vocabulary to map each word to a index
2. Need to setup a PyTorch dataset to load the data
3. Setup padding of every batch (all examples should be of same seq_len and setup dataloader)

Ref:
From walkthrough
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py
"""

import os # for loading filepaths
import pandas as pd #for lookup in annotation file
import spacy # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence #pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image # Load img
import torchvision.transforms as transforms

# Download with: python - m spacy download en
spacy_eng = spacy.load("en")