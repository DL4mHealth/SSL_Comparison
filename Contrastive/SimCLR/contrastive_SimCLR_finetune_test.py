import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from simclr.modules import MLP_Classifier
from utils import yaml_config_hook
import dataset_HAR  # keep this import line although it's in grey
import pandas as pd
from build_dataset import CustomTensorDataset
from simclr import SimCLR_Transformer
from sklearn.metrics import roc_auc_score, average_precision_score, \
    accuracy_score, precision_score, f1_score, recall_score

import warnings
warnings.filterwarnings("ignore")

import random

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_data_loaders_from_arrays(X_train, y_train, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )
    return train_loader


def finetune(args, loader, encoder, classifier, criterion, optimizer, Finetune_mode='Full'):

    print("=== Start Finetuning. ===")
    loss_epoch = []
    accuracy_epoch = []
    precision_epoch = []
    recall_epoch = []
    F1_epoch = []
    auc_epoch = []
    prc_epoch = []
    encoder.train()
    classifier.train()
    for n_batch, (x, x_aug, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        if Finetune_mode == "Full":
            """Full fine-tuning"""
            h, _, z, _ = encoder(x, x)
        else:
            """Partial fine-tuning"""
            with torch.no_grad():
                h, _, z, _ = encoder(x, x)

        output = classifier(h)
        y = y.squeeze(-1).type(torch.LongTensor)
        loss = criterion(output, y)

        predicted = output.argmax(1)

        one_hot_y = F.one_hot(y, num_classes=args.n_class)
        acc = accuracy_score(y, predicted)
        precision = precision_score(y, predicted, average='macro')
        recall = recall_score(y, predicted, average='macro')
        F1 = f1_score(y, predicted, average='macro')
        try:
            auc = roc_auc_score(one_hot_y.squeeze(1).detach().numpy(), output.detach().numpy(), average="macro",
                                multi_class="ovr")
        except:
            auc = np.float(0)
        prc = average_precision_score(one_hot_y.squeeze(1).detach().numpy(), output.detach().numpy(), average="macro")

        accuracy_epoch.append(acc)
        precision_epoch.append(precision)
        recall_epoch.append(recall)
        F1_epoch.append(F1)
        auc_epoch.append(auc)
        prc_epoch.append(prc)

        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())


    mean_loss = sum(loss_epoch)/len(loss_epoch)
    mean_acc_epoch = sum(accuracy_epoch)/len(accuracy_epoch)
    mean_precision_epoch = sum(precision_epoch)/len(precision_epoch)
    mean_recall_epoch = sum(recall_epoch)/len(recall_epoch)
    mean_F1_epoch = sum(F1_epoch)/len(F1_epoch)
    mean_auc_epoch = sum(auc_epoch)/len(auc_epoch)
    mean_prc_epoch = sum(prc_epoch)/len(prc_epoch)

    return mean_loss, mean_acc_epoch, mean_precision_epoch, mean_recall_epoch, \
            mean_F1_epoch, mean_auc_epoch, mean_prc_epoch


def e_test(args, loader, encoder, classifier, criterion):
    # print("=== Start Validation.===")
    loss_epoch = []
    accuracy_epoch = []
    precision_epoch = []
    recall_epoch = []
    F1_epoch = []
    auc_epoch = []
    prc_epoch = []
    encoder.eval()
    classifier.eval()
    for step, (x, x_aug, y) in enumerate(loader):
        encoder.zero_grad()
        classifier.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        h, _, z, _ = encoder(x, x)
        output = classifier(h)
        y = y.squeeze(-1).type(torch.LongTensor)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        one_hot_y = F.one_hot(y, num_classes=args.n_class)
        acc = accuracy_score(y, predicted)
        precision = precision_score(y, predicted, average='macro')
        recall = recall_score(y, predicted, average='macro')
        F1 = f1_score(y, predicted, average='macro')
        try:
            auc = roc_auc_score(one_hot_y.squeeze(1).detach().numpy(), output.detach().numpy(), average="macro",
                                multi_class="ovr")
        except:
            auc = np.float(0)
        prc = average_precision_score(one_hot_y.squeeze(1).detach().numpy(), output.detach().numpy(), average="macro")
        # print(confusion_matrix(y, predicted))

        accuracy_epoch.append(acc)
        precision_epoch.append(precision)
        recall_epoch.append(recall)
        F1_epoch.append(F1)
        auc_epoch.append(auc)
        prc_epoch.append(prc)
        loss_epoch.append(loss.item())

    mean_loss = sum(loss_epoch)/len(loss_epoch)
    mean_acc_epoch = sum(accuracy_epoch)/len(accuracy_epoch)
    mean_precision_epoch = sum(precision_epoch)/len(precision_epoch)
    mean_recall_epoch = sum(recall_epoch)/len(recall_epoch)
    mean_F1_epoch = sum(F1_epoch)/len(F1_epoch)
    mean_auc_epoch = sum(auc_epoch)/len(auc_epoch)
    mean_prc_epoch = sum(prc_epoch)/len(prc_epoch)

    return mean_loss, mean_acc_epoch, mean_precision_epoch, mean_recall_epoch, \
            mean_F1_epoch, mean_auc_epoch, mean_prc_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/HAR_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('We are using %s now.' %args.device)

    setup_seed(args.finetune_seed)

    # load train data
    print("Dataset:", args.dataset)
    dataset_name = globals()['dataset_' + args.dataset]

    """Adjust label ratio, balanced"""
    labelled_ratio = args.labelled_ratio
    fea, lab = [], []
    for i in range(args.n_class):
        print("There are {} samples of Class {}".format((dataset_name.train_y == i).sum(), i))
        ids = dataset_name.train_y == i
        if len(ids.shape) == 1:
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # load val data
    val_dataset = CustomTensorDataset(
        data=(dataset_name.val_x, dataset_name.val_y)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # load test data
    test_dataset = CustomTensorDataset(
        data=(dataset_name.test_x, dataset_name.test_y),
        # transform=Jitterring(0, 0.1),  # No transformation on test data
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # initialize model
    simclr_model = SimCLR_Transformer(
        projection_dim=args.projection_dim,
        n_channel=args.n_channel,
        n_length=args.n_length
    )

    # load pre-trained model from checkpoint
    global arch
    if args.pretrain == True:
        model_fp = os.path.join(args.model_path, "Pretrained_{}_{}_{}.tar".format(
            args.dataset, args.lr, args.projection_dim))
        simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        arch = args.dataset
    else:
        arch = args.dataset + "no_pre"

    simclr_model = simclr_model.to(args.device)

    print("Pre-train:", args.pretrain)

    # Logistic Regression
    n_classes = args.n_class
    classifier = MLP_Classifier(simclr_model.n_features, n_classes)
    classifier = classifier.to(args.device)

    Finetune_mode = "Full"  # or "Partial" "Full"

    if Finetune_mode == "Full":
        optimizer = torch.optim.AdamW([{'params': simclr_model.parameters()},
                                  {'params': classifier.parameters()}], lr=3e-4)
    else:
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # save results
    # finetune_results_df = pd.DataFrame(columns=['Label ratio', 'epoch', 'Loss', 'Acc',
    #                                             'Precision', 'Recall', 'F1', 'AUC', 'PRC'],
    #                                    index=range(0, int((args.logistic_epochs - 1) / 10))
    #                                    )
    # r = 0

    highest_F1 = 0.1
    for epoch in range(args.logistic_epochs):
        mean_loss_train, mean_acc_epoch_train, mean_precision_epoch_train, mean_recall_epoch_train, mean_F1_epoch_train, mean_auc_epoch_train, \
            mean_prc_epoch_train = finetune(
            args, train_loader, simclr_model, classifier, criterion, optimizer, Finetune_mode
        )


        # print(
        #     f"Finetune Epoch [{epoch}/{args.logistic_epochs}]\n Finetune Loss: {mean_loss_train}\n Finetune Accuracy: {mean_acc_epoch_train:.4f}\n"
        #     f"Finetune Precision: {mean_precision_epoch_train:.4f}\n Finetune Recall: {mean_recall_epoch_train:.4f}\n Finetune F1: {mean_F1_epoch_train:.4f}\n"
        #     f"Finetune AUC: {mean_auc_epoch_train:.4f}\n Finetune PRC: {mean_prc_epoch_train:.4f}"
        # )
        # Val
        mean_loss_val, mean_acc_epoch_val, mean_precision_epoch_val, mean_recall_epoch_val, mean_F1_epoch_val, mean_auc_epoch_val, \
            mean_prc_epoch_val  = e_test(
            args, val_loader, simclr_model, classifier, criterion
        ) #val_loader
        # print(
        #      f"Val Epoch [{epoch}/{args.logistic_epochs}]\n Val Loss: {mean_loss_val}\n Val Accuracy: {mean_acc_epoch_val:.4f}\n"
        #      f"Val Precision: {mean_precision_epoch_val:.4f}\n Val Recall: {mean_recall_epoch_val:.4f}\n Val F1: {mean_F1_epoch_val:.4f}\n"
        #      f"Val AUC: {mean_auc_epoch_val:.4f}\n Val PRC: {mean_prc_epoch_val:.4f}"
        # )


        """Update and save model"""

        if mean_F1_epoch_val > highest_F1:
            print('Update fine-tuned model; update highest F1 to {}'.format(mean_F1_epoch_val))
            os.makedirs('save/finetunemodel/', exist_ok=True)
            torch.save(simclr_model.state_dict(), 'save/finetunemodel/' + arch + str(labelled_ratio) + '_model.pt')
            torch.save(classifier.state_dict(), 'save/finetunemodel/' + arch + str(labelled_ratio) + '_classifier.pt')
            highest_F1 = mean_F1_epoch_val



        # Testing
        if epoch % 10 == 0:

            """Load the best model saved above"""
            simclr_model.load_state_dict(torch.load('save/finetunemodel/' + arch + str(labelled_ratio) + '_model.pt'))
            classifier.load_state_dict(torch.load('save/finetunemodel/' + arch + str(labelled_ratio) + '_classifier.pt'))

            print("=== Testing ===")

            mean_loss_test, mean_acc_epoch_test, mean_precision_epoch_test, mean_recall_epoch_test, mean_F1_epoch_test, mean_auc_epoch_test, \
                mean_prc_epoch_test = e_test(
                args, test_loader, simclr_model, classifier, criterion
            )

            print(
                f"Epoch [{epoch}/{args.logistic_epochs}]\n Test Loss: {mean_loss_test}\n Test Accuracy: {mean_acc_epoch_test:.4f}\n"
                f"Test Precision: {mean_precision_epoch_test:.4f}\n Test Recall: {mean_recall_epoch_test:.4f}\n Test F1: {mean_F1_epoch_test:.4f}\n"
                f"Test AUC: {mean_auc_epoch_test:.4f}\n Test PRC: {mean_prc_epoch_test:.4f}"
            )



            # for results
#             finetune_results_df.loc[r] = pd.Series({'Label ratio': args.labelled_ratio, 'epoch': epoch,
#                                                    'Loss': mean_loss_test, 'Acc': mean_acc_epoch_test,
#                                                    'Precision': mean_precision_epoch_test, 'Recall': mean_recall_epoch_test,
#                                                    'F1': mean_F1_epoch_test, 'AUC': mean_auc_epoch_test, 'PRC': mean_prc_epoch_test}
#                                                   )
#             r = r + 1
#
# print(finetune_results_df)
#
# filename = 'SimCLR' + args.dataset + str(args.labelled_ratio) + 'seed' + str(args.finetune_seed) + \
#            'pretrain-' + str(args.pretrain)
# finetune_results_df.to_csv('results/'+filename, sep=',', index=False, encoding='utf-8')