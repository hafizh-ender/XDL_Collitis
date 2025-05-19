import torch
import tqdm
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
          print_every = 10,
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
    loop = tqdm.trange(num_epochs)
    
    for epoch in loop:
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
            targets = torch.tensor([int(t) for t in targets], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            model_outputs_raw = model(data) # Shape: [batch_size, num_classes] still raw probabilities/logits per class
            print(f"model_outputs_raw.shape: {model_outputs_raw.shape}")
            print(f"targets.shape: {targets.shape}")
            print(f"outputs: {model_outputs_raw}")
            print(f"targets: {targets}")
            loss = criterion(y_pred=model_outputs_raw, y_true=targets)
            loss.backward()
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

        # print(f"Memory after epoch: {get_memory_usage()}")
        loop.set_postfix(
            train_loss=train_loss[-1], 
            train_acc=train_acc[-1], 
            val_loss=val_loss[-1], 
            val_acc=val_acc[-1]
        )

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




