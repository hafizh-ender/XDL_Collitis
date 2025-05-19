import torch
import numpy as np
import os
from src.test import test
from src.utils import (
    get_memory_usage,
    clear_memory,
    get_device,
    is_best_model
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
          save_patience = 10,
          save_path = "",
          save_model = True,
          save_metrics = True):
    
    model.to(device)
    
    best_loss = float('inf')
    best_epoch = 0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    train_metrics_per_epoch = {key: np.zeros(num_epochs) for key in metrics.keys()}
    val_metrics_per_epoch = {key: np.zeros(num_epochs) for key in metrics.keys()}
    
    print("Training...")
    # print(f"Initial memory usage: {get_memory_usage()}")
    
    for epoch in range(num_epochs):
        # Clear memory before each epoch
        clear_memory()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        # print(f"Memory before epoch: {get_memory_usage()}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_metrics = {key: np.zeros(len(train_loader)) for key in metrics.keys()}
        val_metrics = {key: np.zeros(len(val_loader)) for key in metrics.keys()}
        
        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            print(f"data.shape: {data.shape}")
            targets = torch.tensor([int(t)-1 for t in targets], dtype=torch.long).to(device)

            # Check for NaNs/Infs in input data
            if torch.isnan(data).any() or torch.isinf(data).any():
                print("!!! NaN or Inf detected in input data !!!")
                # consider raising an error or breaking
            
            optimizer.zero_grad()
            
            model_outputs_raw = model(data) # Shape: [batch_size, num_classes] still raw probabilities/logits per class
            
            # Check for NaNs/Infs in model output
            if torch.isnan(model_outputs_raw).any() or torch.isinf(model_outputs_raw).any():
                print("!!! NaN or Inf detected in model_outputs_raw !!!")
                print(model_outputs_raw)
                # consider raising an error or breaking

            print(f"model_outputs_raw.shape: {model_outputs_raw.shape}")
            print(f"outputs: {model_outputs_raw}")
            print(f"targets: {targets}")
            loss = criterion(y_pred=model_outputs_raw, y_true=targets)
            
            # Check for NaNs/Infs in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("!!! NaN or Inf detected in loss !!!")
                print(f"Loss value: {loss.item()}")
                # consider raising an error or breaking

            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted_indices = model_outputs_raw.argmax(dim=1)
            total += targets.size(0) # targets are the original integer labels
            correct += predicted_indices.eq(targets).sum().item()

            # metrics per batch
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    train_metrics[metric_name][batch_idx] = metric.update(predicted_indices, targets)

            del model_outputs_raw, loss, predicted_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if batch_idx % print_every == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}")
                print(f"Running loss: {running_loss / (batch_idx + 1)}")
                print(f"Metrics: {train_metrics}")
        
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100. * correct / total)
        
        # Validation phase
        model.eval()
        running_val_loss, running_val_acc, val_metrics = test(model, val_loader, device, criterion, metrics, print_every)
        val_loss.append(running_val_loss)
        val_acc.append(running_val_acc)

        if scheduler is not None:
            scheduler.step()

        # metrics per epoch
        for metric_name, metric in metrics.items():
            train_metrics_per_epoch[metric_name][epoch] = train_metrics[metric_name].mean()
            val_metrics_per_epoch[metric_name][epoch] = val_metrics[metric_name].mean()

        # save model
        if not save_model and epoch + 1 < save_patience:
            continue
        if is_best_model(val_loss[-1], best_loss, mode="min"):
            # delete the previous best model
            if os.path.exists(save_path + f"/epoch_{best_epoch+1}.pth"):
                os.remove(save_path + f"/epoch_{best_epoch+1}.pth")
            
            # save new best model
            best_loss = val_loss[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + f"/epoch_{best_epoch+1}.pth")
            print(f"Model saved to {save_path}")

        # Clear memory after each epoch
        clear_memory()
    
    # save metrics
    if save_metrics:
        import json
        with open(save_path + "/metrics.json", "w") as f:
            json.dump(metrics, f)

    return train_loss, train_acc, val_loss, val_acc, train_metrics_per_epoch, val_metrics_per_epoch




