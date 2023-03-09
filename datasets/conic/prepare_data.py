import os
import json

import mmcv
import numpy as np
from skimage import io
from skimage.measure import regionprops

import random

if __name__ == '__main__':
    for phase in ['train', 'val', 'test']:
        mmcv.mkdir_or_exist(f'./{phase}_image')
        mmcv.mkdir_or_exist(f'./{phase}_point')

    images = np.load('./images.npy')
    labels = np.load('./labels.npy')
    classes = ['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective']
    label_dice = dict(zip(range(6), classes))

    num_images = len(images)

    random.seed(0)
    for i in range(num_images):
        ins_map = labels[i, ..., 0]
        cls_map = labels[i, ..., 1]

        p = random.random()
        if p <= 0.2:
            phase = 'val'
        elif p <= 0.4:
            phase = 'test'
        else:
            phase = 'train'

        props = regionprops(ins_map)
        data = dict(zip(classes, [[] for _ in classes]))
        data['classes'] = classes
        for j in range(len(props)):
            y, x = props[j].centroid

            index = cls_map[int(y), int(x)] - 1
            if index >= 0:
                data[label_dice[index]].append([x, y])
            else:
                continue

        io.imsave(f'./{phase}_image/{i}.png', images[i], check_contrast=False)
        with open(f'./{phase}_point/{i}.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(data))

    random.seed(0)
    train_files = os.listdir('./train_image')

    num = len(train_files)
    random.shuffle(train_files)

    for ratio in [5, 10, 15, 20]:
        sup_num = int(num * ratio / 100)
        un_sup_num = num - sup_num

        np.save(f'./labeled_files_{ratio}', train_files[:sup_num])
