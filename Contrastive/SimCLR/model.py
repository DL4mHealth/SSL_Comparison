import os
import torch
import math


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr * args.batch_size / 256,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, args.epochs, eta_min=0, last_epoch=-1
        # )

    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer):
    # out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
    out = os.path.join(args.model_path, "Pretrained_{}_{}_{}.tar".format(
            args.dataset, args.lr, args.projection_dim))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)
