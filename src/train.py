import torch
import numpy as np
import os
import json
from tqdm import tqdm
from src.val import validate
from src.utils import (
    get_memory_usage,
    clear_memory,
    get_device,
    is_best_model,
    is_scheduler_per_batch,
    is_scheduler_requires_val
)

def train(model, 
          train_loader, 
          val_loader, 
          criterion, 
          optimizer, 
          scheduler = None,
          num_epochs = 10, 
          device = get_device(),
          metrics = {},
          print_every = 1,
          early_stopping = 30,
          save_patience = 10,
          save_path = "",
          save_model = True,
          save_metrics = True):
    
    model.to(device)
    if metrics:
        for metric_obj in metrics.values():
            metric_obj.to(device)
    
    best_loss = float('inf')
    best_epoch = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": {key: [] for key in metrics.keys()},
        "val_metrics": {key: [] for key in metrics.keys()}
    }
    
    print("Training...")
    loop = tqdm(range(num_epochs), desc="Epochs")
    for epoch in loop:
        clear_memory()
        # print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        
        if metrics:
            for metric_obj in metrics.values():
                metric_obj.reset()
        # print(len(train_loader)//32) # checking how many loop in train_loader
        index = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            # print(index)
            index += 1
            data = data.to(device)
            targets = targets.to(device)
            # targets = torch.tensor([int(t)-1 for t in targets if str(t).isdigit()], dtype=torch.long).to(device)
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("!!! NaN or Inf detected in input data !!!")
            
            optimizer.zero_grad()
            
            model_outputs_raw = model(data)
            # print(f"targets: {targets}\nmodel_outputs_raw: {model_outputs_raw}")
            
            if torch.isnan(model_outputs_raw).any() or torch.isinf(model_outputs_raw).any():
                print("!!! NaN or Inf detected in model_outputs_raw !!!")
                print(model_outputs_raw)
            
            loss = criterion(input=model_outputs_raw, target=targets)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("!!! NaN or Inf detected in loss !!!")
                print(f"Loss value: {loss.item()}")

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
            
            if metrics:
                for metric_name, metric_obj in metrics.items():
                    if metric_name in ["auroc", "auprc"]:
                        metric_obj.update(model_outputs_raw, target_indices)
                    else:
                        metric_obj.update(predicted_indices, target_indices)

            if is_scheduler_per_batch(scheduler):
                scheduler.step()

            # del model_outputs_raw, loss, predicted_indices
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            # if (batch_idx + 1) % print_every == 0:
            #     print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Train Loss: {running_loss / (batch_idx + 1):.4f}")
        
        
        if not is_scheduler_per_batch(scheduler) and not is_scheduler_requires_val(scheduler):
            scheduler.step()
        
        history["train_loss"].append(running_loss / len(train_loader))
        
        epoch_train_metrics_computed = {}
        if metrics:
            for metric_name, metric_obj in metrics.items():
                val = metric_obj.compute().clone().detach().cpu()
                metric_val_to_store = val.item() if val.numel() == 1 else val.numpy()
                history["train_metrics"][metric_name].append(metric_val_to_store)
                epoch_train_metrics_computed[metric_name] = metric_val_to_store
        
        train_metric_items_str = []
        for name, val in epoch_train_metrics_computed.items():
            if isinstance(val, float):
                train_metric_items_str.append(f"{name}: {val:.4f}")
            elif isinstance(val, np.ndarray):
                formatted_array = np.array2string(val, formatter={'float_kind':lambda x: "%.4f" % x})
                train_metric_items_str.append(f"{name}: {formatted_array}")
            else:
                train_metric_items_str.append(f"{name}: {val}")
        train_metrics_str = ", ".join(train_metric_items_str)
        # print(f"Epoch {epoch+1} Train - Loss: {history['train_loss'][-1]:.4f}, Metrics: {{{train_metrics_str}}}")

        val_loss_epoch, val_metrics_computed = validate(model, val_loader, device, criterion, print_every, metrics)
        history["val_loss"].append(val_loss_epoch)
        
        if metrics:
            for metric_name, value in val_metrics_computed.items():
                if metric_name in history["val_metrics"]:
                    history["val_metrics"][metric_name].append(value)
                else:
                    print(f"Warning: Metric '{metric_name}' from validation not in history init. Value: {value}")
        
        val_metric_items_str = []
        for name, val in val_metrics_computed.items():
            if isinstance(val, float):
                val_metric_items_str.append(f"{name}: {val:.4f}")
            elif isinstance(val, np.ndarray):
                formatted_array = np.array2string(val, formatter={'float_kind':lambda x: "%.4f" % x})
                val_metric_items_str.append(f"{name}: {formatted_array}")
            else:
                val_metric_items_str.append(f"{name}: {val}")
        val_metrics_str = ", ".join(val_metric_items_str)
        # print(f"Epoch {epoch+1} Val - Loss: {val_loss_epoch:.4f}, Metrics: {{{val_metrics_str}}}")

        current_val_loss = history["val_loss"][-1]
        # current_val_loss = history["train_loss"][-1]
        
        if is_scheduler_requires_val(scheduler):
            scheduler.step(current_val_loss)
        
        if save_model and is_best_model(current_val_loss, best_loss, mode="min"):
            if best_epoch > 0 and os.path.exists(os.path.join(save_path, f"epoch_{best_epoch+1}.pth")):
                 try:
                    os.remove(os.path.join(save_path, f"epoch_{best_epoch+1}.pth"))
                 except OSError as e:
                    print(f"Error deleting old model: {e}")
            
            best_loss = current_val_loss
            best_epoch = epoch
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f"epoch_{best_epoch+1}.pth"))
            print(f"Model saved to {os.path.join(save_path, f'epoch_{best_epoch+1}.pth')}")
        elif epoch > best_epoch and epoch + 1 - best_epoch >= early_stopping:
            print(f"Early stopping triggered after {early_stopping} epochs with no improvement since epoch {best_epoch + 1}.")
            break
        loop.set_postfix(train_loss=history["train_loss"][-1], val_loss=history["val_loss"][-1], train_metrics=train_metrics_str, val_metrics=val_metrics_str)
        # clear_memory()
    
    if save_metrics:
        final_history_to_save = {
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
            "train_metrics": {},
            "val_metrics": {}
        }
        if "train_metrics" in history:
            for metric_name, values_list in history["train_metrics"].items():
                final_history_to_save["train_metrics"][metric_name] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in values_list
                ]
        if "val_metrics" in history:
             for metric_name, values_list in history["val_metrics"].items():
                final_history_to_save["val_metrics"][metric_name] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in values_list
                ]
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file_path = os.path.join(save_path, "training_history.json")
        with open(save_file_path, "w") as f:
            json.dump(final_history_to_save, f, indent=4)
        print(f"Training history saved to {save_file_path}")

    return history




