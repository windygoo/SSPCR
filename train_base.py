import sys

import wandb
import argparse

import prettytable as pt
import torch.backends.cudnn as cudnn

from utils import *
from tqdm import tqdm
from eval_map import eval_map
from mmdet.core import multi_apply

from matcher import build_matcher
from dataset import build_dataloader
from criterion import build_criterion
from models import build_model

from torch.nn.parallel import DistributedDataParallel


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * SSOD Settings
    parser.add_argument('--enable_semi_sup', action='store_true')
    parser.add_argument('--burn_up', default=50, type=int)
    parser.add_argument('--ratio', default=10, type=int, choices=[5, 10, 15, 20])
    parser.add_argument('--ema_keep_rate', default=0.99, type=float)

    # * Logger
    parser.add_argument('--use_wandb', action='store_true')

    # * Train
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_eval', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.4, type=float,
                        help="Relative classification weight of the no-object class")

    # * Matcher
    parser.add_argument('--set_cost_point', default=0.1, type=float,
                        help="L2 point coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    # * Model
    parser.add_argument('--space', default=8, type=float)
    parser.add_argument('--num_classes', type=int, default=6,
                        help="Number of cell categories")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the MLPs")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * for CLTR Model
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Dataset
    parser.add_argument('--dataset', default='pdl1', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    # * Evaluator
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--match_dis', default=20, type=int)
    parser.add_argument('--nms_thr', default=-1, type=int)

    # * Distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def do_train():
    rank = args.rank if args.distributed else 0

    if rank == 0 and args.use_wandb:
        run = wandb.init(project='sscr', entity="shuizy")
        run.name = run.id
        run.save()

        cfg = wandb.config
        for k, v in args.__dict__.items():
            setattr(cfg, k, v)

    model = Models(rank)
    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model_without_ddp = model.module

    data_loaders = build_dataloader(args)
    matcher = build_matcher(args)
    criterion = build_criterion(rank, matcher, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load checkpoint
    max_cls_f1 = load_checkpoint(args, model, optimizer) if args.resume else 0

    for epoch in range(args.start_epoch, args.epochs):
        model.train()

        log_info = {}
        log_info = train_one_epoch(model.stu_1, data_loaders['train'], optimizer, epoch, criterion, rank, log_info)

        if epoch >= args.start_eval:
            save_model(epoch, args.output_dir, model, optimizer, max_cls_f1)

            metrics_tea_1, metrics_string_tea_1 = do_eval(model.stu_1,
                                                          data_loaders['val'],
                                                          epoch,
                                                          rank=rank,
                                                          nms_thr=args.nms_thr,
                                                          match_dis=args.match_dis)
            log_info.update(dict(zip(["Tea_1 Det_P", "Tea_1 Det_R", "Tea_1 Det_F1"], metrics_tea_1['Det'])))
            log_info.update(dict(zip(["Tea_1 Cls_P", "Tea_1 Cls_R", "Tea_1 Cls_F1"], metrics_tea_1['Cls'])))
            tea_1_f1 = metrics_tea_1['分类'][-1]

            if rank == 0:
                if tea_1_f1 > max_cls_f1:
                    max_cls_f1 = tea_1_f1
                    save_model(epoch, args.output_dir, model, optimizer, max_cls_f1, metrics_string_tea_1, mode='best')

        if rank == 0 and args.use_wandb:
            wandb.log(log_info, step=epoch)

    if args.distributed:
        cleanup()


def train_one_epoch(student, train_loader, optimizer, epoch, criterion, rank, log_info):
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    iterator = train_loader
    if rank == 0:
        iterator = tqdm(train_loader, file=sys.stdout)
        iterator.set_description(f"Train epoch-{epoch}")

    for data_iter_step, (labeled_images, points, labels, lengths) in enumerate(iterator):
        # warmup lr
        labeled_images = labeled_images.cuda(rank)
        points = points.cuda(rank)
        labels = labels.cuda(rank)

        # all points N×2
        targets = {'gt_nums': lengths,
                   'gt_points': [points_seq[points_seq != -1].reshape(-1, 2) for points_seq in points],
                   'gt_labels': [label_seq[label_seq != -1] for label_seq in labels]}

        outputs = student(labeled_images)
        losses = criterion(outputs, targets, branch='sup')

        optimizer.zero_grad()
        sum(losses.values()).backward()
        optimizer.step()

        gathered_losses = torch.stack(list(losses.values()))
        if args.distributed:
            dist.reduce(gathered_losses, 0, op=dist.ReduceOp.SUM)
            gathered_losses /= args.world_size

        for k, v in zip(losses.keys(), gathered_losses):
            log_info[k] = log_info.get(k, 0) + v.item()

    return log_info


def do_eval(model,
            data_loader_test,
            epoch=0,
            rank=0,
            nms_thr=-1,
            match_dis=20,
            calc_map=False,
            eps=1e-6):
    model.eval()
    class_names = data_loader_test.dataset.classes

    cls_results = []
    cls_annotations = []

    cls_pn, cls_tn = list(torch.zeros(args.num_classes).cuda(rank) for _ in range(2))
    cls_rn = torch.zeros(match_dis, args.num_classes).cuda(rank)

    det_pn, det_tn = list(torch.zeros(1).cuda(rank) for _ in range(2))
    det_rn = torch.zeros(match_dis).cuda(rank)

    iterator = data_loader_test
    if rank == 0:
        iterator = tqdm(data_loader_test, file=sys.stdout)
        iterator.set_description("Test epoch-%d" % epoch)

    for i, (images, gd_points, labels) in enumerate(iterator):
        images = images.cuda(rank)

        pd_points, pd_scores, pd_classes = predict(model, images, nms_thr)

        gd_points = gd_points[0].reshape(-1, 2).numpy()
        labels = labels[0].numpy()

        cls_annotations.append({'bboxes': gd_points, 'labels': labels})

        cls_results_sample = []
        for c in range(args.num_classes):
            ind = (pd_classes == c)
            category_pd_points = pd_points[ind]
            category_gd_points = gd_points[labels == c]

            cls_results_sample.append(np.concatenate([category_pd_points, pd_scores[ind, c][:, None]], axis=-1))

            pred_num, gd_num = len(category_pd_points), len(category_gd_points)
            cls_pn[c] += pred_num
            cls_tn[c] += gd_num

            if pred_num and gd_num:
                cls_right_nums, _ = multi_apply(binary_match,
                                                [category_pd_points],
                                                [category_gd_points],
                                                [match_dis])

                cls_rn[:, c] += torch.tensor(cls_right_nums, device=cls_rn.device)

        cls_results.append(cls_results_sample)

        det_pn += len(pd_points)
        det_tn += len(gd_points)
        if len(pd_points) and len(gd_points):
            det_right_nums, _ = multi_apply(binary_match,
                                            [pd_points],
                                            [gd_points],
                                            [match_dis])

            det_rn += torch.tensor(det_right_nums, device=det_rn.device)

    if args.world_size > 1:
        dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

        dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

    det_r = det_rn / (det_tn + eps) * 100
    det_p = det_rn / (det_pn + eps) * 100
    det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

    cls_r = cls_rn / (cls_tn + eps) * 100
    cls_p = cls_rn / (cls_pn + eps) * 100
    cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

    table = pt.PrettyTable()
    table.add_column('Classes', class_names)

    table.add_column('P', cls_p.mean(0).tolist())
    table.add_column('R', cls_r.mean(0).tolist())
    table.add_column('F1', cls_f1.mean(0).tolist())

    table.add_row(['---'] * 4)

    print(cls_f1.mean(1))

    det_p, det_r, det_f1 = det_p.mean().item(), det_r.mean().item(), det_f1.mean().item()
    cls_p, cls_r, cls_f1 = cls_p.mean().item(), cls_r.mean().item(), cls_f1.mean().item()

    table.add_row(['Det', det_p, det_r, det_f1])
    table.add_row(['Cls', cls_p, cls_r, cls_f1])
    print(table)

    if calc_map:
        eval_map(cls_results, cls_annotations, iou_thr=-match_dis, classes=class_names)

    metrics = {'Det': [det_p, det_r, det_f1], 'Cls': [cls_p, cls_r, cls_f1]}
    return metrics, table.get_string()


@torch.no_grad()
def update_teacher_model(student_model, ema_state_dict, global_step):
    student_model_dict = student_model.state_dict()
    keep_rate = min(1 - 1 / (global_step + 1), args.ema_keep_rate)

    new_teacher_dict = OrderedDict()
    for key, value in ema_state_dict.items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))
    return new_teacher_dict


class Models(nn.Module):
    def __init__(self, rank=0):
        super(Models, self).__init__()

        stu_1 = build_model(args).cuda(rank)
        self.stu_1 = stu_1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    init_distributed_mode(args)
    set_seed(args)
    cudnn.benchmark = True

    if not args.test:

        do_train()

    else:
        from dataset import build_dataset
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        model = Models()

        rank = args.gpu if args.distributed else 0
        ckpt = torch.load(f'./checkpoint/{args.dataset}_sup_{args.ratio}_base/best.pth', map_location='cpu')
        print(ckpt['metrics'], ckpt['epoch'])
        model.load_state_dict(ckpt.get('model', ckpt))
        model.cuda(rank)

        dataset_test = build_dataset(args, 'test')
        test_sampler = DistributedSampler(dataset_test, shuffle=False) if args.distributed else None

        data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, sampler=test_sampler)
        do_eval(model.stu_1, data_loader_test, nms_thr=args.nms_thr, rank=rank, match_dis=args.match_dis)

        if args.distributed:
            cleanup()
