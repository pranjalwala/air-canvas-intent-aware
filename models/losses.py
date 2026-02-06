import torch.nn as nn

# Binary intent loss (TEXT vs DRAW)
intent_criterion = nn.BCEWithLogitsLoss()

