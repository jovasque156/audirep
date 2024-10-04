from .base_models import MLP

class MLPClassifier(MLP):
    def __init__(self, n_features, mlp_layers=[512, 256, 64], p_dropout=None, num_classes=2):
        super(MLPClassifier, self).__init__(n_features, mlp_layers, p_dropout, num_classes)

    def forward(self, x):
        head, x, y = super(MLPClassifier, self).forward(x)
        return head, x, y