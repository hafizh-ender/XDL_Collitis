import torchvision.transforms as transforms

def preprocess(target_input_size, # Expected as (C, H, W)
               rotation_range=None, 
               width_shift_range=None, 
               height_shift_range=None, 
               shear_range=None, 
               zoom_range=None, 
               horizontal_flip=None, 
               vertical_flip=None, 
               brightness_range=None, 
               channel_shift_range=None, 
               fill_mode='nearest'):
    
    if not (isinstance(target_input_size, (list, tuple)) and len(target_input_size) == 3):
        raise ValueError("target_input_size must be a tuple or list of (C, H, W)")
    
    _ , H, W = target_input_size # C is not used for Resize directly
    resize_dims = (H, W)

    # Create transforms that match TensorFlow's ImageDataGenerator functionality
    transform_list = [
        transforms.Resize(resize_dims), # Added Resize
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0) # User's current normalization
    ]
    
    if rotation_range is not None:
        transform_list.append(transforms.RandomRotation(rotation_range))
        
    if width_shift_range is not None or height_shift_range is not None:
        transform_list.append(
            transforms.RandomAffine(0, translate=(width_shift_range, height_shift_range))
        )
        
    if shear_range is not None:
        transform_list.append(transforms.RandomAffine(0, shear=shear_range))
        
    if zoom_range is not None:
        transform_list.append(
            transforms.RandomAffine(0, scale=(1-zoom_range, 1+zoom_range))
        )
        
    if horizontal_flip is not None:
        transform_list.append(transforms.RandomHorizontalFlip())
        
    if vertical_flip is not None:
        transform_list.append(transforms.RandomVerticalFlip())
        
    if brightness_range is not None:
        transform_list.append(
            transforms.ColorJitter(brightness=(0.4, 0.9))
        )
        
    if channel_shift_range is not None:
        # Clamp the channel_shift_range to ensure it falls within the acceptable range
        hue_value = max(min(channel_shift_range/255.0, 0.4), -0.5)
        transform_list.append(
            transforms.ColorJitter(hue=hue_value)
        )
        
        transform_list.append(
            transforms.ColorJitter(saturation=0.4, contrast=0.4)
        )
    
        
    return transforms.Compose(transform_list)