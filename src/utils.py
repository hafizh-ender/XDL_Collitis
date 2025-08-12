import os
import torch
import pandas as pd
import psutil
import gc
import numpy as np
import random
import yaml
from .densenet import DenseNet121

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)\
    
def get_filenames(dataset_path):
    # Get the list of file names
    filenames = os.listdir(dataset_path)
    filenames.sort()

    return filenames

def split_dataset(dataset_dir, class_sources: dict, shuffle = False, seed = 42, split_ratio=[0.8, 0.1, 0.1]):

    full_filenames = {"image_path": [], "class": []}

    for class_label, source_paths in class_sources.items():
        for source_path in source_paths:
            source_dir = os.path.join(dataset_dir, source_path)
            
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory {source_dir} does not exist")
                continue
            
            subdirectories = [d for d in os.listdir(source_dir) 
                            if os.path.isdir(os.path.join(source_dir, d))]
            
            if subdirectories:
                #kalau ga specified, ambil semua
                for subdir in subdirectories:
                    subdir_path = os.path.join(source_dir, subdir)
                    files = get_filenames(subdir_path)
                    
                    for file in files:
                        full_filenames["image_path"].append(os.path.join(subdir_path, file))
                        full_filenames["class"].append(class_label)
            else:
                files = get_filenames(source_dir)
                
                for file in files:
                    full_filenames["image_path"].append(os.path.join(source_dir, file))
                    full_filenames["class"].append(class_label)
    
    full_filenames_df = pd.DataFrame(full_filenames)
    if shuffle:
        full_filenames_df = full_filenames_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_size = int(len(full_filenames_df) * split_ratio[0])
    val_size = int(len(full_filenames_df) * split_ratio[1])

    train_filenames = full_filenames_df.iloc[:train_size]
    val_filenames = full_filenames_df.iloc[train_size:train_size+val_size]
    test_filenames = full_filenames_df.iloc[train_size+val_size:]

    return train_filenames, val_filenames, test_filenames

def generate_filenames_df(dataset_dir, 
                          class_sources: dict, 
                          shuffle=False, 
                          seed=42,
                          is_sample = False,
                          sample_size = 50):
    full_filenames = {"image_path": [], "class": []}

    for class_label, source_paths in class_sources.items():
        for source_path in source_paths:
            source_dir = os.path.join(dataset_dir, source_path)
            
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory {source_dir} does not exist")
                continue
            
            subdirectories = [d for d in os.listdir(source_dir) 
                            if os.path.isdir(os.path.join(source_dir, d))]
            
            if subdirectories:
                #kalau ga specified, ambil semua
                for subdir in subdirectories:
                    subdir_path = os.path.join(source_dir, subdir)
                    files = get_filenames(subdir_path)
                    
                    for file in files:
                        full_filenames["image_path"].append(os.path.join(subdir_path, file))
                        full_filenames["class"].append(class_label)
            else:
                files = get_filenames(source_dir)
                
                for file in files:
                    full_filenames["image_path"].append(os.path.join(source_dir, file))
                    full_filenames["class"].append(class_label)

    full_filenames_df = pd.DataFrame.from_dict(full_filenames)
    if shuffle:
        full_filenames_df = full_filenames_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if is_sample:
        full_filenames_df = full_filenames_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return full_filenames_df

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

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

# traning
def get_training_params(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def is_best_model(current_metric, best_metric, mode="max", monitor_metric="loss"):
    if mode == "max":
        return current_metric > best_metric
    else:
        return current_metric < best_metric

def is_scheduler_per_batch(scheduler):
    if scheduler is None:
        return False
    if (isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
        or isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        or isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
        return True
    else:
        return False

def is_scheduler_requires_val(scheduler):
    if scheduler is None:
        return False
    if (isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        return True
    else:
        return False

def load_model(model_path, num_classes=None, dropout_rate=0.25):
    """
    Load a DenseNet121 model from a saved state dict or create a new one
    
    Args:
        model_path (str): Path to the saved model state dict
        num_classes (int, optional): Number of output classes. Required if creating new model
        dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.25
        
    Returns:
        DenseNet121: Loaded or newly created model
    """
    try:
        # Try to load the model state dict
        state_dict = torch.load(model_path)
        model = DenseNet121(num_classes=num_classes, dropout_rate=dropout_rate)
        model.load_state_dict(state_dict)
        return model
    except:
        # If loading fails, create a new model
        if num_classes is None:
            raise ValueError("num_classes must be specified when creating a new model")
        return DenseNet121(num_classes=num_classes, dropout_rate=dropout_rate)