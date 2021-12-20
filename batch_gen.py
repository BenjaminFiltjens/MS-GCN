import torch
import numpy as np
import random
from data.signals.disps import get_displacements
from data.signals.rel_coords import get_relative_coordinates


def get_features(sample):
    disps = get_displacements(sample)
    rel_coords = get_relative_coordinates(sample)
    sample = np.concatenate([disps, rel_coords], axis=0)
    return sample


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            try:
                string2 = vid[:-10]
                features = np.load(self.features_path + string2 + 'input' + '.npy')
                features = get_features(features)
            except IOError:
                print('stop')
            try:
                file_ptr = np.loadtxt(self.gt_path + vid)
            except ValueError:
                print('stop')
            classes = np.zeros(min(np.shape(features)[1], len(file_ptr)), dtype=int)
            for i in range(len(classes)):
                classes[i] = file_ptr[i].astype(int)
            batch_input.append(features[:, ::self.sample_rate, :, :])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), 6, max(length_of_sequences), 9, 1, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        sample_weight = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1], :, :] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        return batch_input_tensor, batch_target_tensor, mask, sample_weight
