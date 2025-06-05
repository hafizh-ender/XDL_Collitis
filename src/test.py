import torch
from src.utils import clear_memory, get_memory_usage

def test(model, test_loader, device, criterion, print_every=10, metrics=None):
    model.eval()
    test_loss = 0.0
    # Manual accuracy calculation (correct, total) is removed as it should come from metrics if desired.
    
    if metrics:
        for metric in metrics.values():
            metric.to(device)
            metric.reset()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            
            model_outputs_raw = model(data)
            loss = criterion(input = model_outputs_raw, target = targets) # Assuming criterion still uses y_pred, y_true
            test_loss += loss.item()
            
            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
            
            # Update all metrics
            if metrics:
                for metric_name, metric_obj in metrics.items(): # Renamed to metric_obj for clarity
                    if metric_name in ["auroc", "auprc"]:
                        metric_obj.update(model_outputs_raw, target_indices)
                    else:
                        metric_obj.update(predicted_indices, target_indices)

            del model_outputs_raw, loss, predicted_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    test_loss_avg = test_loss / len(test_loader)
    
    computed_metrics = {}
    if metrics:
        for metric_name, metric_obj in metrics.items(): # Renamed to metric_obj
            val = metric_obj.compute().clone().detach().cpu()
            computed_metrics[metric_name] = val.item() if val.numel() == 1 else val.numpy()
            metric_obj.reset()
            
    clear_memory()
    
    # Accuracy, if calculated, will be in computed_metrics under its key (e.g., "accuracy")
    return test_loss_avg, computed_metrics

