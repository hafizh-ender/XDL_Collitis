import os
import torch
import pandas as pd
import psutil
import gc
import numpy as np
import random
import yaml

def get_filenames(dataset_path):
    # Get the list of file names
    filenames = os.listdir(dataset_path)
    filenames.sort()

    return filenames

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def generate_filenames_df(dataset_dir, 
                          categories, 
                          shuffle=False, 
                          seed=42,
                          is_sample = False,
                          sample_size = 50):
    full_filenames = {"image_path": [], "class": []}

    for category in categories:
        full_filenames_temp = get_filenames(os.path.join(dataset_dir, category))

        for i in range(len(full_filenames_temp)):
            full_filenames_temp[i] = os.path.join(dataset_dir, category, full_filenames_temp[i])

        full_filenames["image_path"].extend(full_filenames_temp)
        full_filenames["class"].extend([category] * len(full_filenames_temp))

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

