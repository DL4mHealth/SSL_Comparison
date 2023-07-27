import os
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import random
from model import *
from utils import yaml_config_hook
import pandas as pd
from build_dataset import CustomTensorDataset
import dataset_HAR
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score

import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_avg(result_list):
    return sum(result_list)/len(result_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAE")
    config = yaml_config_hook("config/HAR_config_MAE.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    setup_seed(args.finetune_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", args.device)

    # load data
    print("Dataset:", args.dataset)
    dataset_name = globals()['dataset_' + args.dataset]

    """Adjust label ratio, balanced"""
    labelled_ratio = args.labelled_ratio
    fea, lab = [], []
    for i in range(args.n_class):
        print("There are {} samples of Class {}".format((dataset_name.train_y == i).sum(), i))
        ids = dataset_name.train_y == i
        if len(ids.shape)== 1:
            aa_x = dataset_name.train_x[ids]
            aa_y = dataset_name.train_y[ids]
        else:
            aa_x = dataset_name.train_x[ids.squeeze(1)]
            aa_y = dataset_name.train_y[ids.squeeze(1)]
        n_samples = int(aa_x.shape[0]*labelled_ratio)
        aa_x_short = aa_x[:n_samples]
        aa_y_short = aa_y[:n_samples]
        fea.append(aa_x_short)
        lab.append(aa_y_short)

    fea_flat = np.concatenate(fea, axis=0)
    lab_flat = np.concatenate(lab, axis=0)
    perm = np.random.permutation(fea_flat.shape[0])

    train_x, train_y = fea_flat[perm], lab_flat[perm]

    print("For {} label ratio, {} samples use for fine-tune.".format(labelled_ratio, train_x.shape[0]))

    train_dataset = CustomTensorDataset(
        data=(train_x, train_y),
    )

    # load val data
    val_dataset = CustomTensorDataset(
        data=(dataset_name.val_x, dataset_name.val_y)
    )

    # load test data
    test_dataset = CustomTensorDataset(
        data=(dataset_name.test_x, dataset_name.test_y),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        args.finetune_batch_size,
        shuffle=True,
        drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        args.finetune_batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        args.finetune_batch_size,
        shuffle=True,
        drop_last=True
    )

    model = MAE_ViT(
        sample_shape=[args.n_channel, args.n_length],
        patch_size=(args.n_channel, 10),
        mask_ratio=args.mask_ratio
    )

    global arch
    if args.pretrain == True:
        model_fp = os.path.join(args.model_path, "Pretrained_{}_{}.tar".format(
            args.dataset, args.emb_dim))
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        arch = args.dataset
        print("With pretrain.")
    else:
        arch = args.dataset + "_no_pre"
        print("No pretrain.")

    model = ViT_Classifier(model.encoder, num_classes=args.n_class).to(args.device)

    Finetune_mode = "Full"  # or "Partial"
    if Finetune_mode == "Full":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                      betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.finetune_base_learning_rate * args.finetune_batch_size / 256,
                                      betas=(0.9, 0.999), weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    loss_fn = torch.nn.CrossEntropyLoss()

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    best_val_F1 = 0
    step_count = 0
    optimizer.zero_grad()

    for epoch in range(args.finetune_epochs):
        print("--------start fine-tuning--------")
        model.train()
        loss_epoch = []
        accuracy_epoch = []
        precision_epoch = []
        recall_epoch = []
        F1_epoch = []
        auc_epoch = []
        prc_epoch = []
        for sample, label in tqdm(iter(train_dataloader)):
            step_count += 1
            sample = sample.to(args.device)
            label = label.to(args.device)
            logits = model(sample)
            predicted = logits.argmax(1)
            label = label.type(torch.LongTensor)
            loss = loss_fn(logits, label.squeeze(-1))
            acc = accuracy_score(label, predicted)

            one_hot_y = F.one_hot(label, num_classes=args.n_class)
            precision = precision_score(label, predicted, average='macro')
            recall = recall_score(label, predicted, average='macro')
            F1 = f1_score(label, predicted, average='macro')
            try:
                auc = roc_auc_score(one_hot_y.squeeze(1).detach().numpy(), logits.detach().numpy(), average="macro",
                                    multi_class="ovr")
            except:
                auc = np.float(0)
            prc = average_precision_score(one_hot_y.squeeze(1).detach().numpy(), logits.detach().numpy(),
                                          average="macro")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch.append(loss.item())
            accuracy_epoch.append(acc)
            precision_epoch.append(precision)
            recall_epoch.append(recall)
            F1_epoch.append(F1)
            auc_epoch.append(auc)
            prc_epoch.append(prc)
        lr_scheduler.step()
        avg_train_loss = sum(loss_epoch) / len(loss_epoch)
        avg_train_acc = sum(accuracy_epoch) / len(accuracy_epoch)
        avg_train_precision = sum(precision_epoch)/len(precision_epoch)
        avg_train_recall = sum(recall_epoch)/len(recall_epoch)
        avg_train_F1 = sum(F1_epoch) / len(F1_epoch)
        avg_train_auc = sum(auc_epoch) / len(auc_epoch)
        avg_train_prc = sum(prc_epoch) / len(prc_epoch)
        # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{args.total_epoch}]\n average Finetune Loss: {avg_train_loss}\n Finetune Accuracy: {avg_train_acc:.4f}\n"
                f"Finetune Precision: {avg_train_precision:.4f}\n Finetune Recall: {avg_train_recall:.4f}\n "
                f"Finetune F1: {avg_train_F1:.4f}\n Finetune AUC: {avg_train_auc:.4f}\n Finetune PRC: {avg_train_prc:.4f}"
            )

        model.eval()

        print("--------start validation--------")
        with torch.no_grad():
            loss_epoch = []
            accuracy_epoch = []
            precision_epoch = []
            recall_epoch = []
            F1_epoch = []
            auc_epoch = []
            prc_epoch = []
            for sample, label in tqdm(iter(val_dataloader)):
                sample = sample.to(args.device)
                label = label.to(args.device)
                logits = model(sample)
                predicted = logits.argmax(1)
                label = label.type(torch.LongTensor)

                loss = loss_fn(logits, label.squeeze(-1))
                acc = accuracy_score(label, predicted)
                one_hot_y = F.one_hot(label, num_classes=args.n_class)
                precision = precision_score(label, predicted, average='macro')
                recall = recall_score(label, predicted, average='macro')
                F1 = f1_score(label, predicted, average='macro')
                try:
                    auc = roc_auc_score(one_hot_y.squeeze(1).detach().numpy(), logits.detach().numpy(), average="macro",
                                        multi_class="ovr")
                except:
                    auc = np.float(0)
                prc = average_precision_score(one_hot_y.squeeze(1).detach().numpy(), logits.detach().numpy(),
                                              average="macro")

                loss_epoch.append(loss.item())
                accuracy_epoch.append(acc)
                precision_epoch.append(precision)
                recall_epoch.append(recall)
                F1_epoch.append(F1)
                auc_epoch.append(auc)
                prc_epoch.append(prc)

            avg_val_loss = get_avg(loss_epoch)
            avg_val_acc = get_avg(accuracy_epoch)
            avg_val_precision = get_avg(precision_epoch)
            avg_val_recall = get_avg(recall_epoch)
            avg_val_F1 = get_avg(F1_epoch)
            avg_val_auc = get_avg(auc_epoch)
            avg_val_prc = get_avg(prc_epoch)
            if epoch % 10 == 0:
                print(
                    f"Epoch [{epoch}/{args.finetune_epochs}]\n Average validation Loss: {avg_val_loss}\n Validation Accuracy: {avg_val_acc:.4f}\n"
                    f"Validation Precision: {avg_val_precision:.4f}\n Validation Recall: {avg_val_recall:.4f}\n "
                    f"Validation F1: {avg_val_F1:.4f}\n Validation AUC: {avg_val_auc:.4f}\n Validation PRC: {avg_val_prc:.4f}"
                )

        # use F1 to select best model
        if avg_val_F1 > best_val_F1:
            best_val_F1 = avg_val_F1
            print(f'saving best model with F1 {best_val_F1} at {epoch} epoch!')
            FT_model_path = 'save/finetune/' + arch + str(args.labelled_ratio) + '.pt'
            torch.save(model, FT_model_path)

        if epoch % 10 == 0:
            print("TEST-epoch {}--------start testing-------- ".format(epoch))

            model.eval()
            with torch.no_grad():
                loss_epoch = []
                accuracy_epoch = []
                precision_epoch = []
                recall_epoch = []
                F1_epoch = []
                auc_epoch = []
                prc_epoch = []
                for sample, label in tqdm(iter(test_dataloader)):
                    sample = sample.to(args.device)
                    label = label.to(args.device)
                    logits = model(sample)
                    predicted = logits.argmax(1)
                    label = label.type(torch.LongTensor)

                    loss = loss_fn(logits, label.squeeze(-1))
                    acc = accuracy_score(label, predicted)

                    one_hot_y = F.one_hot(label, num_classes=args.n_class)
                    precision = precision_score(label, predicted, average='macro')
                    recall = recall_score(label, predicted, average='macro')
                    F1 = f1_score(label, predicted, average='macro')
                    try:
                        auc = roc_auc_score(one_hot_y.squeeze(1).detach().numpy(), logits.detach().numpy(), average="macro",
                                            multi_class="ovr")
                    except:
                        auc = np.float(0)
                    prc = average_precision_score(one_hot_y.squeeze(1).detach().numpy(), logits.detach().numpy(),
                                                  average="macro")

                    loss_epoch.append(loss.item())
                    accuracy_epoch.append(acc)
                    precision_epoch.append(precision)
                    recall_epoch.append(recall)
                    F1_epoch.append(F1)
                    auc_epoch.append(auc)
                    prc_epoch.append(prc)
                avg_test_loss = get_avg(loss_epoch)
                avg_test_acc = get_avg(accuracy_epoch)
                avg_test_precision = get_avg(precision_epoch)
                avg_test_recall = get_avg(recall_epoch)
                avg_test_F1 = get_avg(F1_epoch)
                avg_test_auc = get_avg(auc_epoch)
                avg_test_prc = get_avg(prc_epoch)

                print(
                    f"Testing: \n Average test Loss: {avg_test_loss}\n Test Accuracy: {avg_test_acc:.4f}\n"
                    f"Test Precision: {avg_test_precision:.4f}\n Test Recall: {avg_test_recall:.4f}\n "
                    f"Test F1: {avg_test_F1:.4f}\n Test AUC: {avg_test_auc:.4f}\n Test PRC: {avg_test_prc:.4f}"
                )
                print("Pretrain: {}; Label ratio: {}".format(args.pretrain, args.labelled_ratio))
