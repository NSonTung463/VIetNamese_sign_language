import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, inputs):
        global_avg = F.adaptive_avg_pool1d(inputs, 1)
        global_avg = global_avg.unsqueeze(-1)
        eca = self.conv(global_avg)
        eca = eca.squeeze(-1)
        eca = torch.sigmoid(eca)
        eca = eca.unsqueeze(1)
        return inputs * eca

# Example usage:
kernel_size = 5  # Replace with the actual kernel size
eca_layer = ECA(kernel_size)

# Assuming input_tensor has shape (batch_size, channels, sequence_length)
input_tensor = torch.randn(32, 3, 64)  # Replace 32 with your batch size, 3 with the number of channels, and 64 with the sequence length
output = eca_layer(input_tensor)
print(output)
