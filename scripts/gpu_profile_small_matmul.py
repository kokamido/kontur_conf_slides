import torch
torch.backends.cuda.matmul.allow_tf32=True


a = torch.rand((2048,4096), device='cuda',dtype=torch.float32)
b = torch.rand((4096, 2048), device='cuda',dtype=torch.float32)
c = a@b