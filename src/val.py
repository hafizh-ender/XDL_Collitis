import torch
from src.utils import clear_memory, get_memory_usage

def validate(model, val_loader, device, criterion, print_every=10, metrics=None):
    model.eval()
    val_loss = 0.0
    
    if metrics:
        for metric in metrics.values():
            metric.to(device)
            metric.reset()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(device)
            targets = targets.to(device)
            
            model_outputs_raw = model(data)
            loss = criterion(input = model_outputs_raw, target = targets)
            val_loss += loss.item()
            
            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
            
            if metrics:
                for metric_name, metric_obj in metrics.items():
                    if metric_name in ["auroc", "auprc"]:
                        metric_obj.update(model_outputs_raw, target_indices)
                    else:
                        metric_obj.update(predicted_indices, target_indices)

            del model_outputs_raw, loss, predicted_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    val_loss_avg = val_loss / len(val_loader)
    
    computed_metrics = {}
    if metrics:
        for metric_name, metric_obj in metrics.items():
            val = metric_obj.compute().clone().detach().cpu()
            computed_metrics[metric_name] = val.item() if val.numel() == 1 else val.numpy()
            metric_obj.reset()
            
    clear_memory()
    
    return val_loss_avg, computed_metrics

