import numpy as np
import torch
from tqdm import tqdm
def test(model, test_loader, device, verbose=False, print_every=10):
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
            predicted_indices.append(torch.argmax(model_outputs_raw, dim=1).detach().cpu().numpy())
            target_indices.append(torch.argmax(targets, dim=1).detach().cpu().numpy())
            
            if (batch_idx + 1) % print_every == 0 and verbose:
                print(f"Batch {batch_idx + 1}/{len(test_loader)}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predicted_indices_flattened = np.concatenate([np.array(arr) for arr in predicted_indices])
    target_indices_flattened = np.concatenate([np.array(arr) for arr in target_indices])

    return raw_predictions, predicted_indices_flattened, target_indices_flattened