import numpy as np
import torch
from tqdm import tqdm
def test(model, test_loader, device, verbose=False, print_every=10, metrics=None):
    model.eval()
    raw_predictions = []
    target_indices = []
    predicted_indices = []
    loop = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)
            
            model_outputs_raw = model(data)
            raw_predictions.append(model_outputs_raw.detach().cpu())
            predicted_indices.append(torch.argmax(model_outputs_raw, dim=1).detach().cpu())
            target_indices.append(torch.argmax(targets, dim=1).detach().cpu())
            
            if (batch_idx + 1) % print_every == 0 and verbose:
                print(f"Batch {batch_idx + 1}/{len(test_loader)}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate tensors instead of converting to numpy
    raw_predictions_flattened = torch.cat(raw_predictions)
    predicted_indices_flattened = torch.cat(predicted_indices)
    target_indices_flattened = torch.cat(target_indices)
    
    if metrics:
        test_metrics = {}
        for metric_name, metric_obj in metrics.items():
            if metric_name in ["auroc", "auprc"]:
                # For binary classification metrics, use the positive class probabilities
                positive_class_probs = raw_predictions_flattened[:, 1]  # Get probabilities for class 1
                metric_obj.update(positive_class_probs, target_indices_flattened)
            else:
                metric_obj.update(predicted_indices_flattened, target_indices_flattened)
            val = metric_obj.compute().clone().detach().cpu()
            metric_val_to_store = val.item() if val.numel() == 1 else val.numpy()
            test_metrics[metric_name] = metric_val_to_store
    else:
        test_metrics = None

    # Convert to numpy only at the end for return values
    predicted_indices_flattened = predicted_indices_flattened.numpy()
    target_indices_flattened = target_indices_flattened.numpy()

    return raw_predictions, predicted_indices_flattened, target_indices_flattened, test_metrics