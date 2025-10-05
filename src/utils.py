import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(train_dataset, device):
    y_train = [s[1] for s in train_dataset.samples]
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    return torch.tensor(class_weights, dtype=torch.float).to(device)
