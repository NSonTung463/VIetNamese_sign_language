import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomTransformerEncoder(nn.Module):
    def __init__(self, n_landmark, d_model, num_layers, num_classes, num_heads, dim_feedforward):
        super(CustomTransformerEncoder, self).__init__()
        self.embedding =  nn.Linear(n_landmark, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,  dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        transformer_output = self.transformer_encoder(x)
        mean_pooled = transformer_output.mean(dim=1)
        logits = self.fc(mean_pooled)
        return logits

if __name__ == "__main__":
    from torch.autograd import Variable
    
    # Thông số của mô hình {} embed_dim must be divisible by num_heads
    n_landmark = 390
    d_model = 256  
    dim_feedforward = 256
    num_layers = 2
    num_heads = 2
    num_classes = 9
    
    # Xây dựng mô hình
    model = CustomTransformerEncoder(n_landmark, d_model, num_layers, num_classes, num_heads, dim_feedforward)
    # In thông tin về kiến trúc mô hình
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    x = torch.rand((4,124,390))
    print(model(x).shape)