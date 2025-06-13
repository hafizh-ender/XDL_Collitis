import os
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import cv2
import numpy as np
import matplotlib.pyplot as plt

def smoothgrad(model, input_tensor, target_class, n_samples=100, noise_level=0.05):
    """
    Compute SmoothGrad saliency map
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target_class: Target class index
        n_samples: Number of noisy samples to average (default: 100)
        noise_level: Standard deviation of noise to add (default: 0.05)
        
    Returns:
        SmoothGrad saliency map
    """
    model.eval()
    input_tensor.requires_grad = True
    
    # Initialize gradient accumulation
    accumulated_gradients = torch.zeros_like(input_tensor)
    
    # Calculate input range for noise scaling
    input_range = torch.max(input_tensor) - torch.min(input_tensor)
    scaled_noise_level = noise_level * input_range
    
    for _ in range(n_samples):
        # Add scaled noise to input
        noise = torch.randn_like(input_tensor) * scaled_noise_level
        noisy_input = input_tensor + noise
        
        # Forward pass
        output = model(noisy_input)
        
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
            
        # Zero gradients
        model.zero_grad()
        
        # Compute gradients
        loss = output[0, target_class]
        loss.backward()
        
        # Accumulate gradients
        if input_tensor.grad is not None:
            accumulated_gradients += input_tensor.grad.data
            input_tensor.grad.data.zero_()
    
    # Average gradients
    smoothgrad_map = accumulated_gradients / n_samples
    
    # Take absolute value and normalize
    smoothgrad_map = torch.abs(smoothgrad_map)
    smoothgrad_map = (smoothgrad_map - smoothgrad_map.min()) / (smoothgrad_map.max() - smoothgrad_map.min() + 1e-8)
    
    return smoothgrad_map[0].cpu().numpy()

def plot_XDL_Visualizations(model, test_loader, device, num_samples=5, print_img=False, print_every=10, save_path=None):
    model.eval()
    target_layer = model.densenet_model.features.denseblock4
    categories = test_loader.dataset.categories
    
    # Initialize GradCAM
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )
    
    samples_processed = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        if samples_processed >= num_samples:
            break
            
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            model_outputs_raw = model(data)
            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
        
        for i in range(min(len(data), num_samples - samples_processed)):
            img = data[i].cpu().numpy()
            pred_idx = predicted_indices[i].item()
            true_idx = target_indices[i].item()
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            
            # Get GradCAM visualization
            grayscale_cam = cam(
                input_tensor=data[i:i+1],
                targets=[ClassifierOutputTarget(pred_idx)]
            )[0, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            # Get SmoothGrad visualization
            smoothgrad_map = smoothgrad(model, data[i:i+1], pred_idx)
            smoothgrad_map = np.transpose(smoothgrad_map, (1, 2, 0))
            smoothgrad_map = np.mean(smoothgrad_map, axis=2)  # Convert to grayscale
            smoothgrad_map = cv2.resize(smoothgrad_map, (img.shape[1], img.shape[0]))
            smoothgrad_map = np.uint8(255 * smoothgrad_map)
            smoothgrad_map = cv2.applyColorMap(smoothgrad_map, cv2.COLORMAP_JET)
            smoothgrad_map = cv2.addWeighted(np.uint8(255 * img), 0.6, smoothgrad_map, 0.4, 0)
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            if print_img:
                ax1.imshow(img)
            image_path = test_loader.dataset.dataframe.iloc[samples_processed]['image_path']
            ax1.set_title(f'Original Image: {os.path.join(*image_path.split("/")[-3:])}')
            ax1.axis('off')
            
            if print_img:
                ax2.imshow(cam_image)
            ax2.set_title('GradCAM Visualization')
            ax2.axis('off')
            
            if print_img:
                ax3.imshow(smoothgrad_map)
            ax3.set_title('SmoothGrad Visualization')
            ax3.axis('off')
            
            plt.tight_layout()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'{samples_processed}.png'))
                plt.close()
            elif print_img:
                plt.show()
            else:
                plt.close()
            samples_processed += 1
            
        if batch_idx % print_every == 0:
            print(f'Processed {samples_processed} samples')

# Keep the original plot_XDL_GradCAM function for backward compatibility
def plot_XDL_GradCAM(model, test_loader, device, num_samples=5, print_img=False, print_every=10, save_path=None):
    model.eval()
    target_layer = model.densenet_model.features.denseblock4
    categories = test_loader.dataset.categories
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )
    samples_processed = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        if samples_processed >= num_samples:
            break
            
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            model_outputs_raw = model(data)
            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
        
        for i in range(min(len(data), num_samples - samples_processed)):
            img = data[i].cpu().numpy()
            pred_idx = predicted_indices[i].item()
            true_idx = target_indices[i].item()
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            
            grayscale_cam = cam(
                input_tensor=data[i:i+1],
                targets=[ClassifierOutputTarget(pred_idx)]
            )[0, :]
            
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            if print_img:
                ax1.imshow(img)
            image_path = test_loader.dataset.dataframe.iloc[samples_processed]['image_path']
            ax1.set_title(f'Original Image: {os.path.join(*image_path.split("/")[-3:])}\nTrue: {categories[true_idx]}, Pred: {categories[pred_idx]}')
            ax1.axis('off')
            
            if print_img:
                ax2.imshow(cam_image)
            ax2.set_title(f'GradCAM Visualization: {os.path.join(*image_path.split("/")[-3:])}')
            ax2.axis('off')
            
            plt.tight_layout()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'{samples_processed}.png'))
                plt.close()
            elif print_img:
                plt.show()
            else:
                plt.close()
            samples_processed += 1
        if batch_idx % print_every == 0:
            print(f'Processed {samples_processed} samples')

def plot_XDL_SmoothGrad(model, 
                         test_loader, 
                         device, 
                         num_samples=5, 
                         print_img=True,
                         print_every=10, 
                         save_path=None, 
                         n_samples_smoothgrad=100,
                         noise_level=0.05):
    """
    Plot SmoothGrad visualizations for model predictions
    """
    model.eval()
    categories = test_loader.dataset.categories
    samples_processed = 0
    
    for batch_idx, (data, targets) in enumerate(test_loader):
        if samples_processed >= num_samples:
            break
            
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            model_outputs_raw = model(data)
            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
        
        for i in range(min(len(data), num_samples - samples_processed)):
            img = data[i].cpu().numpy()
            pred_idx = predicted_indices[i].item()
            true_idx = target_indices[i].item()
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            
            # Get SmoothGrad visualization
            smoothgrad_map = smoothgrad(model, data[i:i+1], pred_idx, n_samples=n_samples_smoothgrad, noise_level=noise_level)
            smoothgrad_map = np.transpose(smoothgrad_map, (1, 2, 0))
            smoothgrad_map = np.mean(smoothgrad_map, axis=2)  # Convert to grayscale
            smoothgrad_map = cv2.resize(smoothgrad_map, (img.shape[1], img.shape[0]))
            
            # Improve visualization
            smoothgrad_map = (smoothgrad_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(smoothgrad_map, cv2.COLORMAP_VIRIDIS)  # Changed to VIRIDIS colormap
            
            # Adjust overlay weights for better visibility
            overlay = cv2.addWeighted(np.uint8(255 * img), 0.7, heatmap, 0.3, 0)  # More weight to original image
            
            # Create visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            image_path = test_loader.dataset.dataframe.iloc[samples_processed]['image_path']
            plt.title(f'Original Image: {os.path.join(*image_path.split("/")[-3:])}\nTrue: {categories[true_idx]}, Pred: {categories[pred_idx]}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(overlay)
            plt.title('SmoothGrad Visualization')
            plt.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f'smoothgrad_{samples_processed}.png'))
            
            if print_img:
                plt.show()
            else:
                plt.close()
                
            samples_processed += 1
            
        if batch_idx % print_every == 0:
            print(f'Processed {samples_processed} samples')