import torch
import tqdm

def train(model, 
          train_loader, 
          val_loader, 
          criterion, 
          optimizer, 
          num_epochs, 
          device):
    
    print("Training...")
    loop = tqdm.trange(num_epochs)
    for epoch in loop:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data = data.to(device)
                targets = targets.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total     
        
        loop.set_postfix(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)




