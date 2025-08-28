import os
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.densenet_multilayer import DenseNet121Multilayer

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
    accumulated_gradients = torch.zeros_like(input_tensor)
    input_range = torch.max(input_tensor) - torch.min(input_tensor)
    scaled_noise_level = noise_level * input_range

    for _ in range(n_samples):
        # Clone and detach input for each sample
        noisy_input = input_tensor + torch.randn_like(input_tensor) * scaled_noise_level
        noisy_input = noisy_input.clone().detach().requires_grad_(True)

        output_1, output_2 = model(noisy_input)
        # if len(output.shape) == 1:
        #     output = output.unsqueeze(0)

        model.zero_grad()
        loss = output_1[0, target_class]
        loss.backward()

        if noisy_input.grad is not None:
            accumulated_gradients += noisy_input.grad.data

    smoothgrad_map = accumulated_gradients / n_samples
    smoothgrad_map = torch.abs(smoothgrad_map)
    smoothgrad_map = (smoothgrad_map - smoothgrad_map.min()) / (smoothgrad_map.max() - smoothgrad_map.min() + 1e-8)
    return smoothgrad_map[0].cpu().numpy()

def plot_XDL_Visualizations(model: DenseNet121Multilayer, test_loader, device, num_samples=5, print_img=False, print_every=10, save_path=None, 
                          smoothgrad_percentile=95, smoothgrad_colormap='hot', smoothgrad_overlay_alpha=0.5):
    """
    Plot visualizations with GradCAM and SmoothGrad. batch size has to be 1 !!!!
    """
    model.eval()

    print(f"Using model: {model.__class__.__name__}")

    target_layer_1 = model.model_layer_2_1.densenet_model.features[-1]
    target_layer_2 = model.model_layer_2_2.densenet_model.features[-1]
    categories = test_loader.dataset.categories
    
    # Initialize GradCAM
    cam = [GradCAM(model=model,
        target_layers=[target_layer_1]),
        GradCAM(model=model,
        target_layers=[target_layer_2])]

    samples_processed = 0

    samples_processed_per_class = {i: 0 for i in range(len(categories))}
    max_samples_per_class = num_samples // len(categories) if num_samples > len(categories) else num_samples

    for batch_idx, (data, targets) in enumerate(test_loader):
        
        if samples_processed >= num_samples:
            break

            
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            model_outputs_raw = model(data)
            first_predicted_indices = torch.argmax(model.out_1, dim=1)

            predicted_indices = torch.argmax(model_outputs_raw, dim=1)
            target_indices = torch.argmax(targets, dim=1)
        
        # Check if we have enough samples for each class
        for i in range(len(categories)):
            if samples_processed_per_class[i] >= max_samples_per_class:
                continue

        print(f'Processing batch {batch_idx + 1}, samples processed: {samples_processed}')
        
        for i in range(min(len(data), num_samples - samples_processed)):
            img = data[i].cpu().numpy()
            pred_idx = predicted_indices[i].item()
            true_idx = target_indices[i].item()

            print(f'Processing sample {samples_processed + 1}, raw prediction: {model_outputs_raw[i].cpu().numpy()}, predicted index: {pred_idx}, true index: {true_idx}')
            
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Get GradCAM visualization
            grayscale_cam = cam[first_predicted_indices](
                input_tensor=data[i:i+1],
                targets=[ClassifierOutputTarget(pred_idx)]
            )[0, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
            # Get SmoothGrad visualization
            smoothgrad_map = smoothgrad(model, data[i:i+1], pred_idx)
            smoothgrad_map = np.transpose(smoothgrad_map, (1, 2, 0))
            smoothgrad_map = np.mean(smoothgrad_map, axis=2)  # Convert to grayscale
            smoothgrad_map = cv2.resize(smoothgrad_map, (img.shape[1], img.shape[0]))
            
            # Increase contrast by clipping at specified percentile and re-normalizing
            p_clip = np.percentile(smoothgrad_map, smoothgrad_percentile)
            if p_clip > 0:
                smoothgrad_map = np.clip(smoothgrad_map, 0, p_clip)
                smoothgrad_map = (smoothgrad_map - smoothgrad_map.min()) / (p_clip - smoothgrad_map.min() + 1e-8)

            # Improve visualization
            smoothgrad_heatmap = (smoothgrad_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(smoothgrad_heatmap, cv2.COLORMAP_HOT if smoothgrad_colormap == 'hot' else cv2.COLORMAP_VIRIDIS)
            
            # Adjust overlay weights for better visibility
            overlay = cv2.addWeighted(img_uint8, 1 - smoothgrad_overlay_alpha, heatmap, smoothgrad_overlay_alpha, 0)
            
            # Create visualization
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
            
            # Firs row, First column: original image with label information
            axes[0, 0].imshow(img)
            axes[0, 0].set_title(f'True: {categories[true_idx]}, Pred: {categories[pred_idx]}\nConfidence: {torch.nn.functional.softmax(model_outputs_raw[i], dim=0)[pred_idx]:.4f}')
            axes[0, 0].axis('off')

            # First row, Second column: GradCAM visualization
            axes[0, 1].imshow(cam_image)
            axes[0, 1].set_title('GradCAM Visualization')
            axes[0, 1].axis('off')

            # Add colorbar for GradCAM
            norm_cam = colors.Normalize(vmin=grayscale_cam.min(), vmax=grayscale_cam.max())
            sm_cam = plt.cm.ScalarMappable(cmap='jet', norm=norm_cam)
            sm_cam.set_array([])
            fig.colorbar(sm_cam, ax=axes[0, 1], shrink=0.8)

            # First row, Third column: SmoothGrad visualization
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title('SmoothGrad Visualization')
            axes[0, 2].axis('off')
            
            # Add colorbar for SmoothGrad
            norm_smooth = colors.Normalize(vmin=smoothgrad_map.min(), vmax=smoothgrad_map.max())
            sm_smooth = plt.cm.ScalarMappable(cmap='viridis', norm=norm_smooth)
            sm_smooth.set_array([])
            fig.colorbar(sm_smooth, ax=axes[0, 2], shrink=0.8)
            
            # Second row, First column: original image with filter (only show pixels if SmoothGrad value above threshold)
            threshold = 0.5
            
            # Add channel dimension if needed
            smoothgrad_map_copy = smoothgrad_map.copy()
            if len(smoothgrad_map_copy.shape) == 2:
                smoothgrad_map_copy = np.expand_dims(smoothgrad_map_copy, axis=-1)
                smoothgrad_map_copy = np.repeat(smoothgrad_map_copy, repeats=3, axis=-1)

            filtered_img = np.where(smoothgrad_map_copy > threshold, img, 1)
            axes[1, 0].imshow(filtered_img)
            axes[1, 0].set_title(f'Filtered Image (SmoothGrad > {threshold})')
            axes[1, 0].axis('off')
            
            # Second row, Third column: SmoothGrad heatmap
            axes[1, 2].imshow(smoothgrad_map, cmap='viridis')
            axes[1, 2].set_title('SmoothGrad Heatmap')
            axes[1, 2].axis('off')
            
            # Add colorbar for SmoothGrad heatmap
            sm_smooth_heatmap = plt.cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=smoothgrad_map.min(), vmax=smoothgrad_map.max()))
            sm_smooth_heatmap.set_array([])
            fig.colorbar(sm_smooth_heatmap, ax=axes[1, 2], shrink=0.8)
            
            # Second row, Second column: original image with filter (only show pixels if GradCAM value above threshold)
            cam_threshold = 0.5
            
            # Add channel dimension if needed
            cam_image_copy = grayscale_cam.copy()
            if len(cam_image_copy.shape) == 2:
                cam_image_copy = np.expand_dims(cam_image_copy, axis=-1)
                cam_image_copy = np.repeat(cam_image_copy, repeats=3, axis=-1)
            filtered_cam_img = np.where(cam_image_copy > cam_threshold, img, 1)
            axes[1, 1].imshow(filtered_cam_img)
            axes[1, 1].set_title(f'Filtered Image (GradCAM > {cam_threshold})')
            axes[1, 1].axis('off')
            
            # Add colorbar (to adjust spacing) but make it invisible
            norm_cam_invis = colors.Normalize(vmin=0, vmax=0)
            sm_cam_invis = plt.cm.ScalarMappable(cmap='Greys', norm=norm_cam_invis)
            sm_cam_invis.set_array([])
            fig.colorbar(sm_cam_invis, ax=axes[1, 1], shrink=0.8)

            plt.tight_layout()

            if save_path:
                true_label = categories[true_idx]
                pred_label = categories[pred_idx]
                os.makedirs(save_path, exist_ok=True)
                
                if pred_label == true_label:
                    plt.savefig(os.path.join(save_path, f'{true_label}_{samples_processed}.png'))
                else:
                    plt.savefig(os.path.join(save_path, f'wrong_{true_label}_{samples_processed}.png'))

            if print_img:
                plt.show()

            plt.close(fig)
            samples_processed += 1
            
        if batch_idx % print_every == 0:
            print(f'Processed {samples_processed} samples')