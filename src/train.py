import torch
import tqdm
import psutil
import gc
import os

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024  # in MB
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # in MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # in MB
        return {
            'cpu_memory': f"{cpu_memory:.2f}MB",
            'gpu_memory_allocated': f"{gpu_memory:.2f}MB",
            'gpu_memory_reserved': f"{gpu_memory_reserved:.2f}MB"
        }
    return {'cpu_memory': f"{cpu_memory:.2f}MB"}

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train(model, 
          train_loader, 
          val_loader, 
          criterion, 
          optimizer, 
          num_epochs, 
          device):
    
    print("Training...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    loop = tqdm.trange(num_epochs)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    for epoch in loop:
        # Clear memory before each epoch
        clear_memory()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Memory before epoch: {get_memory_usage()}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Clear memory every 10 batches
            if batch_idx % 10 == 0:
                clear_memory()
                print(f"Memory at batch {batch_idx}: {get_memory_usage()}")
            
            data = data.to(device)
            targets = torch.tensor([int(t) for t in targets], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Clear some memory after each batch
            del outputs, loss, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100. * correct / total)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Memory before validation: {get_memory_usage()}")
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                # Clear memory every 10 batches
                if batch_idx % 10 == 0:
                    clear_memory()
                    print(f"Memory at validation batch {batch_idx}: {get_memory_usage()}")
                
                data = data.to(device)
                targets = torch.tensor([int(t) for t in targets], dtype=torch.long).to(device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Clear some memory after each batch
                del outputs, loss, predicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        val_loss.append(running_loss / len(val_loader))
        val_acc.append(100. * correct / total)
        
        print(f"Memory after epoch: {get_memory_usage()}")
        loop.set_postfix(
            train_loss=train_loss[-1], 
            train_acc=train_acc[-1], 
            val_loss=val_loss[-1], 
            val_acc=val_acc[-1]
        )
        
        # Clear memory after each epoch
        clear_memory()
    
    return train_loss, train_acc, val_loss, val_acc




