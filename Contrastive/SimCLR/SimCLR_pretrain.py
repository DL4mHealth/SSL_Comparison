import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from simclr import SimCLR_Transformer
from simclr.modules import NT_Xent
from model import load_optimizer, save_model
from utils import yaml_config_hook
import time
import torch.utils.data as Data

from build_dataset import CustomTensorDataset
import dataset_HAR
from simclr.modules.transformations import Jittering, Scaling, Flipping



def train(args, train_loader, model, criterion, optimizer):         # writer):
    loss_epoch = []
    for step, (x_i, x_j, _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        """If you are using cuda, uncomment the following 2 lines while comment the 2 mps lines"""
        # x_i = x_i.cuda(non_blocking=True)
        # x_j = x_j.cuda(non_blocking=True)
        """For mps or cpu"""
        x_i = x_i.to(args.device)  # remove .cuda() and change to this when using mps
        x_j = x_j.to(args.device)

        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        loss_epoch.append(loss.item())

    mean_loss = sum(loss_epoch)/len(loss_epoch)
    return mean_loss


def main(gpu, args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Dataset:", args.dataset)
    dataset_name = globals()['dataset_' + args.dataset]

    train_dataset = CustomTensorDataset(
        data=(dataset_name.train_x, dataset_name.train_y),
        transform_A=Jittering(0, 0.1),
        # transform_B=Jittering(0, 0.1),
        # transform_A=Scaling(),
        # transform_B=Flipping(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # initialize model
    model = SimCLR_Transformer(
        projection_dim=args.projection_dim,
        n_channel=args.n_channel,
        n_length=args.n_length
    )

    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature)

    model = model.to(args.device)

    # args.global_step = 0
    args.current_epoch = 0
    print("Training started.")
    lowest_loss = 10
    for epoch in range(args.epochs):
        start_time = time.time()

        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer)

        scheduler.step()

        if loss_epoch < lowest_loss:
            print('Update saved model; update lowest loss to {}'.format(loss_epoch))
            save_model(args, model, optimizer)
            lowest_loss = loss_epoch

        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}\t lr: {round(lr, 8)}"
        )
        print("--- %.4s seconds ---" % (time.time() - start_time))
        args.current_epoch += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("config/HAR_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()  # load all the hyper-para


    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    print(args.device)

    main(0, args)

