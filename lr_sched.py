import math
import copy


def adjust_learning_rate(args, optimizer, glo_epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    lr = args.lr
    # if glo_epoch < args.warmup_epochs:
    #     _lr = lr * glo_epoch / args.warmup_epochs
    # else:
    #     _lr = args.min_lr + (lr - args.min_lr) * 0.5 * \
    #           (1. + math.cos(math.pi * (glo_epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    _lr = args.min_lr + (lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * glo_epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if "lr_decay" in param_group:
            param_group["lr"] = _lr * param_group["lr_decay"]
        else:
            param_group["lr"] = _lr


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:  # absence of number
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks."):].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def build_param_group(args, model):
    base_lr = args.lr
    weight_decay = args.weight_decay

    params = []
    memo = set()

    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay

    overrides = {'pos_embed': {'weight_decay': 0.0}}
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            hyperparams["lr_decay"] = get_vit_lr_decay_rate(f"{module_name}.{module_param_name}", num_layers=12,
                                                            lr_decay_rate=0.7)
            hyperparams["lr"] *= hyperparams["lr_decay"]
            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})

    return params


if __name__ == '__main__':
    lr = 1e-4
    min_lr = 1e-5

    epochs = 100
    for glo_epoch in range(101):
        _lr = min_lr + (lr - min_lr) * 0.5 * (1. + math.cos(math.pi * glo_epoch / epochs))
        print(_lr)
