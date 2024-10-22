import torch
import torch.nn.functional as F
import torch.nn as nn


class PartialLoss(nn.Module):
    def __init__(self, confidence, partialY, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.partialY = partialY
        self.conf_ema_m = conf_ema_m

    def forward(self, outputs, index):
        average_loss = F.mse_loss(F.softmax(outputs, dim=-1), self.confidence[index, :])
        return average_loss

    def confidence_update_soft(self, pseudo_label, batch_index):
        with torch.no_grad():
            self.confidence[batch_index, :] = torch.nn.functional.normalize(
                self.conf_ema_m * self.confidence[batch_index, :]
                + (1 - self.conf_ema_m) * pseudo_label, p=1, dim=-1)


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, device, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask, batch_size=-1):

        mask = mask.float().detach().to(self.device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def celoss(outputs, partialy):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * partialy
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss
