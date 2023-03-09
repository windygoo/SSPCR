import torch
from torch import nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, num_classes, matcher, class_weight, loss_weight):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def loss_reg(self, outputs, targets, indices, num_points, branch='sup'):
        """ Regression loss """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_coords'][idx]

        target_points = torch.cat([gt_points[J] for gt_points, (_, J) in zip(targets['gt_points'], indices)], dim=0)

        loss_pnt = F.mse_loss(src_points, target_points, reduction='none')
        # loss_pnt = F.l1_loss(src_points, target_points, reduction='none')

        loss_reg = loss_pnt.sum() / (num_points + 1e-7)
        return loss_reg

    def loss_cls(self, outputs, targets, indices, num_points, branch='sup'):
        """Classification loss """
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits']
        device = src_logits.device

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.long, device=device)
        target_classes_o = torch.cat([cls[J] for cls, (_, J) in zip(targets['gt_labels'], indices)])
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)

        return loss_cls

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, use_for_reg=None, branch='sup'):
        indices = self.matcher(outputs, targets)

        num_points = sum(targets['gt_nums'])
        num_points = torch.as_tensor(num_points, dtype=torch.float)

        losses = {}
        if branch == 'sup':
            loss_map = {
                'loss_reg': self.loss_reg,
                'loss_cls': self.loss_cls
            }
        else:
            loss_map = {
                'loss_cls': self.loss_cls,
            }

        for loss_name, loss_func in loss_map.items():
            weight = 1.0 if branch == 'un_sup' else 1
            losses[f'{branch}_{loss_name}'] = weight * loss_func(outputs, targets, indices, num_points, branch)

        weight_dict = self.loss_weight
        for k in losses:
            losses[k] *= weight_dict[k[len(branch) + 1:]]

        return losses


def build_criterion(rank, matcher, args):
    class_weight = torch.ones(args.num_classes + 1, dtype=torch.float, device=f'cuda:{rank}')
    class_weight[-1] = args.eos_coef
    loss_weight = {
        'loss_reg': args.reg_loss_coef,
        'loss_cls': args.cls_loss_coef
    }

    criterion = Criterion(args.num_classes,
                          matcher,
                          class_weight=class_weight,
                          loss_weight=loss_weight)
    return criterion
