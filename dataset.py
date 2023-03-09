import os
import sys
import albumentations as A

from tqdm import tqdm
from skimage import io
from transforms import *

from utils import collate_fn_pad, read_from_json

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def img_loader(dataset, num_classes, phase):
    keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, num_classes)]
    img_dir, pnt_dir = f'./datasets/{dataset}/{phase}_image', f'./datasets/{dataset}/{phase}_point'
    data, files, classes = [[] for _ in range(3)]
    # if phase == 'val':  # for faster evaluation
    #     reader = tqdm(os.listdir(img_dir)[:200], file=sys.stdout)
    # else:
    #     reader = tqdm(os.listdir(img_dir), file=sys.stdout)
    reader = tqdm(os.listdir(img_dir), file=sys.stdout)
    reader.set_description(f"loading {phase} data")
    for file in reader:
        files.append(file)
        values = [io.imread(os.path.join(img_dir, file))]
        annotations = read_from_json(f"./datasets/{dataset}/{phase}_point/{os.path.splitext(file)[0]}.json")
        values += [np.array(annotations[c]).reshape(-1, 2) for c in annotations['classes']]
        if not classes:
            classes = annotations['classes']
            assert num_classes == len(classes)
        data.append(dict(zip(keys, values)))

    return data, files, classes


class DataFolder(Dataset):
    def __init__(self, dataset, num_classes, phase, data_transform, ratio=5):
        self.phase = phase
        self.data, self.files, self.classes = img_loader(dataset, num_classes, phase)
        self.data_transform = data_transform

        # self.available_data = []
        # self.external_data = []

        available_data = []
        external_data = []
        if phase == 'train':
            labeled_files = np.load(f'./datasets/{dataset}/labeled_files_{ratio}.npy')
            for sample, file in zip(self.data, self.files):
                if file in labeled_files:
                    available_data.append(sample)
                else:
                    external_data.append(sample)

            self.data = available_data
            self.files = labeled_files
            self.external_data = external_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        index = index % len(self.data)
        mini_batch_data = self.data[index]
        mini_batch_data = self.data_transform(mini_batch_data)

        return mini_batch_data


def build_dataset(args, image_set):
    if image_set == 'train':
        additional_targets = {}
        for i in range(1, args.num_classes):
            additional_targets.update({'keypoints%d' % i: 'keypoints'})
        augmentor = A.Compose([
            # A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=0),
            A.RandomGridShuffle(grid=(4, 4), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
        ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
        transform = Preprocessing(augmentor=augmentor)
    else:
        transform = Preprocessing()

    data_folder = DataFolder(args.dataset, args.num_classes, image_set, transform, args.ratio)
    return data_folder


def build_dataloader(args):
    dataset_train = build_dataset(args, 'train')
    dataset_val = build_dataset(args, 'val')
    dataset_test = build_dataset(args, 'test')

    train_sampler = DistributedSampler(dataset_train) if args.distributed else None
    val_sampler = DistributedSampler(dataset_val) if args.distributed else None
    test_sampler = DistributedSampler(dataset_test) if args.distributed else None

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False if train_sampler else True,
                                   sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate_fn_pad)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=val_sampler, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.num_workers)

    data_loaders = {'train': data_loader_train, 'val': data_loader_val, 'test': data_loader_test}
    return data_loaders
