import os
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_XDL_GradCAM(model, test_loader, device, num_samples=5, print_img = False,print_every=10, save_path=None):
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
            ax1.set_title(f'Original Image: {image_path}\nTrue: {categories[true_idx]}, Pred: {categories[pred_idx]}')
            ax1.axis('off')
            
            if print_img:
                ax2.imshow(cam_image)
            ax2.set_title(f'GradCAM Visualization: {image_path}')
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