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
          num_epochs, 
          device = get_device(),
          metrics = {},
          print_every = 10,
          save_patience = 10,
          save_path = "",
          save_model = True,
          save_metrics = True):
    
    print(device)
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
    print(f"Initial memory usage: {get_memory_usage()}")
    loop = tqdm.trange(num_epochs)
    
    for epoch in loop:
        # Clear memory before each epoch
        clear_memory()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Memory before epoch: {get_memory_usage()}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_metrics = {key: np.zeros(len(train_loader)) for key in metrics.keys()}
        val_metrics = {key: np.zeros(len(val_loader)) for key in metrics.keys()}
        
        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = torch.tensor([int(t) for t in targets], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            print(f"outputs: {outputs}")
            print(f"outputs[0].shape: {outputs[0].shape}")
            print(f"targets: {targets}")

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # metrics per batch
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    train_metrics[metric_name][batch_idx] = metric.update(outputs, targets)

            # Clear some memory after each batch
            del outputs, loss, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100. * correct / total)
        
        # Validation phase
        model.eval()
        running_val_loss, running_val_acc, val_metrics = test(model, val_loader, device, criterion, metrics, print_every)
        val_loss.append(running_val_loss)
        val_acc.append(running_val_acc)

        # metrics per epoch
        for metric_name, metric in metrics.items():
            train_metrics_per_epoch[metric_name][epoch] = train_metrics[metric_name].mean()
            val_metrics_per_epoch[metric_name][epoch] = val_metrics[metric_name].mean()

        print(f"Memory after epoch: {get_memory_usage()}")
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




