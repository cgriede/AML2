import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_properties(0))
print(torch.cuda.get_device_properties(0).total_memory)