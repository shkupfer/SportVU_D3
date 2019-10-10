import pickle
from torch.utils.data import Dataset
import torch
import logging
import sys
import time
import torch
import os
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use CPU or GPU
device = torch.device("cpu")

class PossessionsDataset(Dataset):
    def __init__(self, pklfile_names, targets=False, rearrange=False, transform=None):
        print("Initializing PossessionsDataset")
        self.coords_data = []
        self.poss_data = []
        self.targets = []
        self.transform = transform
        n_files = len(pklfile_names)
        for n, pklfile_path in enumerate(pklfile_names):
            if n % 100 == 0:
                print("Loading possession file %s of %s" % (n + 1, n_files))
            with open(pklfile_path, 'rb') as pklfile:
                from_pkl = pickle.load(pklfile)

            coords_data = torch.stack((from_pkl['data'][:, 0, :, :], from_pkl['data'][:, 2, :, :]), dim=1)
            if rearrange:
                coords_data = coords_data.view(coords_data.size(1), coords_data.size(0), coords_data.size(2), coords_data.size(3))
            self.coords_data.append(coords_data)
            # self.poss_data.append(from_pkl['poss_data'])
            # self.targets.append(from_pkl['target'])
        print("Loaded all .pkl files")

        # self.data = torch.stack(self.data)
        # self.targets = torch.stack(self.targets)
        # print("Combined .pkl files into tensors: %s, %s" % (str(self.data.size()), str(self.targets.size())))

    def __len__(self):
        return len(self.coords_data)

    def __getitem__(self, idx):
        # return self.coords_data[idx], self.poss_data[idx], self.targets[idx]
        return self.coords_data[idx]


class OnLoadPossessionsDataset(Dataset):
    def __init__(self, pklfile_names, targets=False, playerchans=False, rearrange=False, transform=None, target_transform=None):
        print("Initializing OnLoadPossessionsDataset")
        self.pklfile_names = pklfile_names
        self.rearrange = rearrange
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        self.playerchans = playerchans

    def __len__(self):
        return len(self.pklfile_names)

    def __getitem__(self, idx):
        with open(self.pklfile_names[idx], 'rb') as pklfile:
            from_pkl = pickle.load(pklfile)
        if self.playerchans:
            coords_data = from_pkl['data']
        else:
            coords_data = torch.stack((from_pkl['data'][:, 0, :, :], from_pkl['data'][:, 2, :, :]), dim=1)

        if self.rearrange:
            coords_data = coords_data.view(coords_data.size(1), coords_data.size(0), coords_data.size(2), coords_data.size(3))
        if self.transform:
            coords_data = torch.stack([self.transform(timestep_coords) for timestep_coords in coords_data])

        if self.targets:
            target = from_pkl['target']
            # if self.target_transform:
            return coords_data, target
        else:
            return coords_data


class PossessionsDatasetTorch(Dataset):
    def __init__(self, coords_data_fnames, target_fnames=None, targets=False, playerchans=False, rearrange=False, transform=None, target_transform=None):
        print("Initializing PossessionsDatasetTorch")
        self.target_fnames = target_fnames
        self.coords_data_fnames = coords_data_fnames
        self.rearrange = rearrange
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        self.playerchans = playerchans

    def __len__(self):
        return len(self.target_fnames)

    def __getitem__(self, idx):
        coords_data = torch.load(self.coords_data_fnames[idx])
        if not self.playerchans:
            coords_data = torch.stack((coords_data[:, 0, :, :], coords_data[:, 2, :, :]), dim=1)

        if self.rearrange:
            coords_data = coords_data.view(coords_data.size(1), coords_data.size(0), coords_data.size(2), coords_data.size(3))
        if self.transform:
            coords_data = torch.stack([self.transform(timestep_coords) for timestep_coords in coords_data])

        if self.targets:
            target = torch.load(self.target_fnames[idx])
            # if self.target_transform:
            return coords_data, target
        else:
            return coords_data


class Simple2DPossSet(Dataset):
    def __init__(self, data_filenames, targets=False, inds=False, transform=None, off_ball_only=True):
        self.transform = transform
        self.data_fnames = data_filenames
        self.targets = targets
        self.off_ball_only = off_ball_only
        self.inds = inds

    def __len__(self):
        return len(self.data_fnames)

    def __getitem__(self, idx):
        with open(self.data_fnames[idx], 'rb') as pklfile:
            from_pkl = pickle.load(pklfile)

        coords_data = from_pkl['data']
        if self.off_ball_only:
            coords_data = coords_data[:6]
        if self.transform:
            coords_data = self.transform(coords_data)
        if self.targets:
            target = from_pkl['target']
            return coords_data, target
        elif self.inds:
            return coords_data, idx
        else:
            return coords_data


class ExtractedFeatures(Dataset):
    def __init__(self, data_filename, moments_per_file, targetsfile=None):
        arr = np.load(data_filename)
        tens = torch.Tensor(arr)
        tens = tens.view(tens.size(0), moments_per_file, -1)
        self.possessions = tens
        self.targetsfile = targetsfile
        if targetsfile:
            with open(targetsfile, 'r') as tgtfile:
                self.targets = torch.Tensor([int(t) for t in tgtfile.read().strip().split()])

    def __len__(self):
        return len(self.possessions)

    def __getitem__(self, idx):
        if self.targetsfile:
            return self.possessions[idx], self.targets[idx]
        else:
            return self.possessions[idx]


class PadSequence:
    def __init__(self, targets=False):
        self.targets = targets

    def __call__(self, batch):
        if self.targets:
            # logger.info("Top of PadSequence.__call__()")
            sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
            sequences = [x[0] for x in sorted_batch]
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            lengths = torch.LongTensor([len(x) for x in sequences])
            targets = torch.Tensor([[x[1]] for x in sorted_batch])
            # logger.info("Bottom of PadSequence.__call__()")
            return sequences_padded.to(device), lengths.to(device), targets.to(device)
        else:
            sequences = sorted(batch, key=lambda x: x.size(0), reverse=True)
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            lengths = torch.LongTensor([len(x) for x in sequences])
            return sequences_padded.to(device), lengths.to(device)
