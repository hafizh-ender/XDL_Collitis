import torch
from src.utils import clear_memory, get_memory_usage

def test(model, test_loader, device, criterion, print_every=10, metrics=None, test_metrics={}):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            
            data = data.to(device)
            targets = torch.tensor([int(t) for t in targets], dtype=torch.long).to(device)
            
            model_outputs_raw = model(data) # Shape: [batch_size, num_classes]
            
            loss = criterion(y_pred=model_outputs_raw, y_true=targets)
            
            test_loss += loss.item()
            
            predicted_indices = model_outputs_raw.argmax(dim=1)
            
            total += targets.size(0)
            correct += predicted_indices.eq(targets).sum().item()

            # metrics
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    test_metrics[metric_name][batch_idx] = metric.update(predicted_indices, targets)

            del model_outputs_raw, loss, predicted_indices # Adjusted variable names
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, test_metrics

