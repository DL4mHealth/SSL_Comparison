import torch.nn as nn


class MLP_Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP_Classifier, self).__init__()

        n_dim = int(int(n_features/n_classes)/2)*n_classes

        self.model = nn.Sequential(
            nn.Linear(n_features, n_dim),
            nn.BatchNorm1d(n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, n_classes)
        )

    def forward(self, x):
        return self.model(x)
