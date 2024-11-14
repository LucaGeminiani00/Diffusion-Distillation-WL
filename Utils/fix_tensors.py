import torch


def clean_keys(t_keys, numsteps):
    changeable = []
    for element in t_keys:
        if t_keys[element].shape == torch.Size([numsteps]): 
            changeable.append(element)
    return changeable

def reshape_target(original, target_size):
    if target_size < 2:
        raise ValueError("Target size must be at least 2 to include first and last elements.")
    
    reshaped_tensor = torch.empty(target_size)
    reshaped_tensor[0] = original[0]  
    reshaped_tensor[-1] = original[-1]  

    num_middle_elements = target_size - 2  
    if num_middle_elements > 0:
        middle_indices = torch.randperm(len(original) - 2)[:num_middle_elements] + 1  
        reshaped_tensor[1:-1] = original[middle_indices]  

    return reshaped_tensor

def reshape_tensors(teacher_keys, changeable, target = None):
    if target is None: 
        for element in changeable: 
            original = teacher_keys[element]
            length = original.shape[0]

            if length % 2 != 0:
                reshaped_tensor = original[:length - 1:2]
            else:
                reshaped_tensor = original[::2]

            teacher_keys[element] = reshaped_tensor
    else:
        for element in changeable: 
            original = teacher_keys[element]
            reshaped_tensor = reshape_target(original, target)
            teacher_keys[element] = reshaped_tensor
