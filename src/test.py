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
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # metrics
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    test_metrics[metric_name][batch_idx] = metric.update(outputs, targets)

            # Clear some memory after each batch
            del outputs, loss, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, test_metrics

