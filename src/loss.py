import torch
import torch.nn as nn


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, targets, model_output):
        # Handle tuple output from model
        if isinstance(model_output, tuple):
            _, digit_probs = model_output
        else:
            digit_probs = model_output
            
        # Convert targets to one-hot encoding if needed
        if len(targets.shape) == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=digit_probs.shape[1])
            
        present_losses = (
            targets * torch.clamp_min(self.m_pos - digit_probs, min=0.0) ** 2
        )
        absent_losses = (1 - targets) * torch.clamp_min(
            digit_probs - self.m_neg, min=0.0
        ) ** 2
        losses = present_losses + self.lambda_ * absent_losses
        return torch.mean(torch.sum(losses, dim=1))


class ReconstructionLoss(nn.Module):
    def forward(self, reconstructions, input_images):
        return torch.nn.MSELoss(reduction="mean")(reconstructions, input_images)


class TotalLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5, recon_factor=0.0005):
        super(TotalLoss, self).__init__()
        self.margin_loss = MarginLoss(m_pos, m_neg, lambda_)
        self.recon_loss = ReconstructionLoss()
        self.recon_factor = recon_factor

    def forward(self, input_images, targets, reconstructions, digit_probs):
        margin = self.margin_loss(targets, digit_probs)
        recon = self.recon_loss(reconstructions, input_images)
        return margin + self.recon_factor * recon