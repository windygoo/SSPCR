import torch
import numpy as np
from torchvision import transforms


class Preprocessing(object):
    def __init__(self,
                 means=(0.485, 0.456, 0.406),
                 stds=(0.229, 0.224, 0.225),
                 augmentor=None):
        self.augmentor = augmentor

        self.trans = transforms.Compose([transforms.ToTensor()])
        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

    def __call__(self, sample: dict):
        if self.augmentor is not None:
            sample = self.augmentor(**sample)
            del sample['transform_params']
            # transform_params = imgs.pop('transform_params')

        data = list(sample.values())
        labels = torch.tensor([], dtype=torch.long)

        for i in range(len(data)):
            if not len(data[i]):
                data[i] = torch.zeros(1, 0, 2, dtype=torch.int)
            else:
                data[i] = self.trans(np.array(data[i]))

            if i == 0:  # image standardization
                for t, m, s in zip(data[0], self.means, self.stds):
                    t.sub_(m).div_(s)
            else:  # reshape to 2N-length vector for padding
                labels = torch.cat([labels, torch.full((data[i].shape[1],), i - 1)])
                data[i] = data[i].reshape(-1)

        return data[0], torch.cat(data[1:]), labels
