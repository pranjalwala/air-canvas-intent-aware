import torch
import torch.nn.functional as F

from models.losses import (
    intent_criterion,
    ctc_criterion,
    contrastive_loss,
    motion_smoothness_loss,
)


def compute_losses(
    model_outputs,
    batch,
    weights=None,
):
    """
    Explicit intent-gated loss computation.

    Args:
        model_outputs (dict): output of AirCanvasModel
        batch (dict): contains ground-truth data
        weights (dict): optional loss weights

    Returns:
        dict of individual losses + total loss
    """

    if weights is None:
        weights = {
            "intent": 1.0,
            "text": 1.0,
            "semantic": 1.0,
            "smooth": 1.0,
        }

    # -------------------------
    # Unpack model outputs
    # -------------------------
    intent_logits = model_outputs["intent_logits"]      # (B, 1)
    text_log_probs = model_outputs["text_log_probs"]    # (T, B, V)
    drawing_emb = model_outputs["drawing_emb"]          # (B, E)

    # -------------------------
    # Unpack batch
    # -------------------------
    intent_targets = batch["intent"]                    # (B,)
    motion = batch["motion"]                            # (B, T, N, d)

    # TEXT supervision
    targets = batch["text_targets"]                     # (sum L,)
    target_lengths = batch["text_target_lengths"]       # (B,)
    input_lengths = batch["text_input_lengths"]         # (B,)

    # DRAW supervision
    text_emb = batch["semantic_text_emb"]               # (B, E)

    # -------------------------
    # Intent loss (always active)
    # -------------------------
    intent_targets = intent_targets.float().unsqueeze(1)
    intent_loss = intent_criterion(intent_logits, intent_targets)

    # -------------------------
    # Intent probabilities for gating
    # -------------------------
    intent_prob = torch.sigmoid(intent_logits).squeeze(1)   # (B,)
    text_gate = intent_prob                                 # TEXT = 1
    draw_gate = 1.0 - intent_prob                           # DRAW = 0

    # -------------------------
    # Text loss (CTC)
    # -------------------------
    text_loss_raw = ctc_criterion(
        text_log_probs,
        targets,
        input_lengths,
        target_lengths,
    )

    text_loss = (text_gate.mean()) * text_loss_raw

    # -------------------------
    # Semantic (drawing) loss
    # -------------------------
    semantic_loss_raw = contrastive_loss(drawing_emb, text_emb)
    semantic_loss = (draw_gate.mean()) * semantic_loss_raw

    # -------------------------
    # Motion smoothness (always active)
    # -------------------------
    smooth_loss = motion_smoothness_loss(motion)

    # -------------------------
    # Total loss
    # -------------------------
    total_loss = (
        weights["intent"] * intent_loss
        + weights["text"] * text_loss
        + weights["semantic"] * semantic_loss
        + weights["smooth"] * smooth_loss
    )

    return {
        "total": total_loss,
        "intent": intent_loss.detach(),
        "text": text_loss.detach(),
        "semantic": semantic_loss.detach(),
        "smooth": smooth_loss.detach(),
    }
