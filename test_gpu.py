import torch


def test_gpu(tensor, gpu_id):
    print(f"tensor.device: {tensor.device}")
    print(f"gpu_id: {gpu_id}")
    tensor2 = torch.empty((1000, 1000)).to(f'cuda:{gpu_id}')
    print(f"tensor.device: {tensor.device}")
    tensor3 = tensor[234, 543] + tensor2[134, 549]
    print(f"tensor3.device: {tensor3.device}")

for device_id in range(7):
    for i in range(1000):
        tensor = torch.empty((1000, 1000)).to(f'cuda:{device_id}')
        test_gpu(tensor, device_id)
