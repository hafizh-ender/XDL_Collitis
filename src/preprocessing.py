import torchvision.transforms as transforms

def preprocess(rotation_range, 
               width_shift_range, 
               height_shift_range, 
               shear_range, 
               zoom_range, 
               horizontal_flip, 
               vertical_flip, 
               brightness_range, 
               channel_shift_range, 
               fill_mode, 
               drop_out):
    
    # Create transforms that match TensorFlow's ImageDataGenerator functionality
    transform_list = [transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))]  # rescale equivalent
    
    if rotation_range:
        transform_list.append(transforms.RandomRotation(rotation_range))
        
    if width_shift_range or height_shift_range:
        transform_list.append(
            transforms.RandomAffine(0, translate=(width_shift_range, height_shift_range))
        )
        
    if shear_range:
        transform_list.append(transforms.RandomAffine(0, shear=shear_range))
        
    if zoom_range:
        transform_list.append(
            transforms.RandomAffine(0, scale=(1-zoom_range, 1+zoom_range))
        )
        
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
        
    if vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())
        
    if brightness_range:
        transform_list.append(
            transforms.ColorJitter(brightness=brightness_range[1]-brightness_range[0])
        )
        
    if channel_shift_range:
        transform_list.append(
            transforms.ColorJitter(hue=channel_shift_range/255.0)
        )
        
    if drop_out:
        transform_list.append(transforms.RandomErasing())
        
    return transforms.Compose(transform_list)