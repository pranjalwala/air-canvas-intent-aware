import torch.nn as nn

# Intent loss
intent_criterion = nn.BCEWithLogitsLoss()

# Text recognition loss (CTC)
ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
