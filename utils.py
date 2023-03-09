import os
import json

import mmcv
import random

import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np
import scipy.spatial as S

from datetime import datetime
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from torch.nn.utils.rnn import pad_sequence


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def cleanup():
    dist.destroy_process_group()


def binary_match(pred_points, gd_points, thr=12):
    dis = S.distance_matrix(pred_points, gd_points)
    connection = np.zeros_like(dis)
    connection[dis <= thr] = 1
    graph = csr_matrix(connection)
    res = maximum_bipartite_matching(graph, perm_type='column')
    right_points_index = np.where(res > 0)[0]
    right_num = len(right_points_index)

    return right_num, right_points_index


def point_nms(points, scores, nms_thr):
    _reserved = np.ones(len(points), dtype=bool)
    dis_matrix = S.distance_matrix(points, points)
    np.fill_diagonal(dis_matrix, np.inf)

    for idx in np.argsort(-scores.max(1)):
        if _reserved[idx]:
            _reserved[dis_matrix[idx] <= nms_thr] = False

    points = points[_reserved]
    classes = scores[_reserved].argmax(-1)
    scores = scores[_reserved]

    return points, scores, classes


def load_checkpoint(args, model, optimizer1, optimizer2=None):
    checkpoint = torch.load(f'./checkpoint/{args.resume}/latest.pth', map_location='cpu')

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)
    optimizer1.load_state_dict(checkpoint['optimizer1'])

    if optimizer2 is not None:
        optimizer2.load_state_dict(checkpoint['optimizer2'])
    args.start_epoch = checkpoint['epoch'] + 1

    return checkpoint.get('cls_mf1', 0)


def save_model(epoch, output_dir, model, optimizer1, cls_mf1, metrics_string='', mode='latest', optimizer2=None):
    if output_dir == '':
        output_dir = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mmcv.mkdir_or_exist(f'./checkpoint/{output_dir}')

    torch.save({
        'epoch': epoch,
        'cls_mf1': cls_mf1,
        'model': model.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict() if optimizer2 else None,
        'metrics': metrics_string
    }, f'./checkpoint/{output_dir}/{mode}.pth')


def set_seed(args):
    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate_fn_pad(batch):
    batch.sort(key=lambda x: len(x[2]), reverse=True)  # sort by the number of points
    labeled_images, points, labels, lengths = [[] for _ in range(4)]
    for x in batch:
        labeled_images.append(x[0])
        points.append(x[1])
        labels.append(x[2])
        lengths.append(len(x[2]))

    points = pad_sequence(points, batch_first=True, padding_value=-1).reshape(len(batch), -1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1).reshape(len(batch), -1)

    return torch.stack(labeled_images), points.float(), labels.long(), lengths


@torch.no_grad()
def predict(model,
            images,
            nms_thr=-1):
    h, w = images.shape[-2:]
    outputs = model(images)

    points = outputs['pred_coords'][0].cpu().numpy()
    scores = outputs['pred_logits'][0].softmax(-1).cpu().numpy()

    cross_border_flag = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
    points = points[~cross_border_flag]
    scores = scores[~cross_border_flag]

    classes = np.argmax(scores, axis=-1)
    _reserved = (classes < (scores.shape[-1] - 1))

    points = points[_reserved]
    scores = scores[_reserved]
    classes = classes[_reserved]

    if len(points) and nms_thr > 0:
        points, scores, classes = point_nms(points, scores, nms_thr=nms_thr)

    return points, scores, classes


def read_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data

# if __name__ == '__main__':
#     import cv2 as cv
#     from skimage import io
#     import albumentations as A
#
#     # 标注数据: 1353
#     # 未标注数据: 1771
#
#     additional_targets = {}
#     for i in range(1, 6):
#         additional_targets.update({'keypoints%d' % i: 'keypoints'})
#     # augmentor = A.Compose([
#     #     A.RandomGridShuffle(grid=(4, 4), p=0.5),
#     #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=0.5),
#     #     A.VerticalFlip(p=0.5),
#     #     A.HorizontalFlip(p=0.5)
#     # ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
#     augmentor = A.Compose([
#         A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
#         A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5)
#     ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
#     image = io.imread('./datasets/her2/train_image/1771.png')
#
#     data = read_from_json('./datasets/her2/train_point/1771.json')
#     keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, 6)]
#     values = [image, ]
#     values += [np.array(data[c]).reshape(-1, 2) for c in data['classes']]
#
#     for i in range(10):
#         results = augmentor(**dict(zip(keys, values)))
#         aug_img = results['image']
#         io.imsave(f'strong_aug_img_{i}.png', aug_img)
#
#     # colors = [(255, 193, 193), (252, 121, 21), (61, 144, 31), (255, 0, 0), (153, 0, 254), (0, 38, 255)]
#     # for i, key in enumerate(keys[1:]):
#     #     for (x, y) in np.array(results[key], dtype=int):
#     #         cv.circle(aug_img, (x, y), 6, colors[i], -1, lineType=cv.LINE_AA)
#
#     # for i, c in enumerate(data['classes']):
#     #     for (x, y) in np.array(data[c], dtype=int):
#     #         cv.circle(image, (x, y), 6, colors[i], -1, lineType=cv.LINE_AA)
#     #
#     # io.imsave('ori_img.png', image)
#
#     # image = io.imread('./datasets/her2/train_image/1771.png')
#     #
#     # # image = cv.flip(image, 1)
#     # additional_targets = {}
#     # for i in range(1, 6):
#     #     additional_targets.update({'keypoints%d' % i: 'keypoints'})
#     # strong_aug = A.Compose([
#     #     A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
#     #     A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5)
#     # ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
#     #
#     # io.imsave('flip_img.png', image)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"

    # TSML [   942. 179562.  80127.  15826.   1467.  84985.]
    # TSML + LM [  2133. 172911.  79490.  19034.   2059.  84625.]
    # TSML + LM + CT tea1 [  2420. 171886.  83340.  19837.   2115.  83647.]
    # TSML + LM + CT tea2 [  2286. 172889.  83858.  19563.   2078.  82659.]
    # GT [2904. 161712.  69631.  18515.   2201.  73160.]

    gt = np.array([2904, 161712, 69631, 18515, 2201, 73160])
    gt = gt / gt.sum()
    gt_log = np.log(gt)

    ser1 = np.array([942, 179562, 80127, 15826, 1467, 84985])
    ser1 = ser1 / ser1.sum()
    ser1_log = np.log(ser1)

    ser2 = np.array([2133, 172911, 79490, 19034, 2059, 84625])
    ser2 = ser2 / ser2.sum()
    ser2_log = np.log(ser2)

    ser3 = np.array([2420, 171886, 83340, 19837, 2115, 83647])
    ser3 = ser3 / ser3.sum()
    ser3_log = np.log(ser3)

    import scipy.stats
    print(scipy.stats.entropy(ser1, gt))
    print(scipy.stats.entropy(ser2, gt))
    print(scipy.stats.entropy(ser3, gt))

    plt.plot(gt_log, label='GT')
    plt.plot(ser1_log, label='TSML')
    plt.plot(ser2_log, label='TSML + LRD')
    plt.plot(ser3_log, label='TSML + LRD + CT')

    plt.legend()
    plt.show()
