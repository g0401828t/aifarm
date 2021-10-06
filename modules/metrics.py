"""
"""
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

def get_metric_fn(y_pred, y_answer, y_prob):
    """ Metric 함수 반환하는 함수

    Returns:
        metric_fn (Callable)
    """
    assert len(y_pred) == len(y_answer), 'The size of prediction and answer are not same.'
    accuracy = accuracy_score(y_answer, y_pred)
    return accuracy

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=1, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            F_loss