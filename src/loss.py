import torch
import torch.nn as nn

def marginLoss(y_true, y_pred):
    # y_true: integer tensor of shape [batch_size] (true class labels)
    # y_pred: float tensor of shape [batch_size, num_classes] (model's raw output/scores)
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1

    # Determine number of classes from y_pred
    num_classes = y_pred.shape[1]

    # Convert y_true (integer labels) to one-hot encoding
    y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=num_classes).float()
    # Ensure y_true_one_hot is on the same device as y_pred
    y_true_one_hot = y_true_one_hot.to(y_pred.device)

    L = y_true_one_hot * torch.clamp(m_plus - y_pred, min=0.0).pow(2) + \
        lbd * (1 - y_true_one_hot) * torch.clamp(y_pred - m_minus, min=0.0).pow(2)

    return torch.mean(torch.sum(L, dim=1)) # Sum over classes, then mean over batch

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
        # if len(targets.shape) == 1:
        #     targets = torch.nn.functional.one_hot(targets, num_classes=3)
            
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