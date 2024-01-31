import torch
from torch import nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_HE_UNIFORM = torch.nn.init.kaiming_uniform_
INIT_GLOROT_UNIFORM = torch.nn.init.xavier_uniform_
INIT_ZEROS = torch.nn.init.zeros_

class LandmarkEmbedding(nn.Module):
    def __init__(self, units,n_columns):
        super(LandmarkEmbedding, self).__init__()

        # Embedding for missing landmark in frame, initialized with zeros
        self.empty_embedding = nn.Parameter(torch.zeros(units), requires_grad=False)
        self.dense_1 = nn.Linear(in_features=n_columns, out_features=units, bias=False, dtype=torch.float32)

        # Embedding
        self.dense = nn.Sequential(
            nn.Linear(in_features=units, out_features=units, bias=False, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(in_features=units, out_features=units, bias=False, dtype=torch.float32),
        )
        self.dense.apply(self.init_weights)

    #init weight
    def init_weights(self, m):
        if type(m) == nn.Linear:
            if m.weight.requires_grad:
                if m in self.dense:
                    INIT_GLOROT_UNIFORM(m.weight)
                else:
                    INIT_HE_UNIFORM(m.weight)

    def forward(self, x):
        # Checks whether landmark is missing in frame
        mask = torch.sum(x, dim=2, keepdim=True) == 0

        # If so, the empty embedding is used
        x = self.dense_1(x)
        empty_embedding_expanded = self.empty_embedding.expand(x.shape)

        # Otherwise the landmark data is embedded
        dense_output = self.dense(x)

        return torch.where(mask, empty_embedding_expanded, dense_output)

class Embedding(nn.Module):
    def __init__(self,N_TARGET_FRAMES,units_embedding,N_COLUMNS,MEANS,STDS):
        super(Embedding, self).__init__()
        self.N_COLUMNS = N_COLUMNS
        self.MEANS = MEANS
        self.STDS= STDS
        self.positional_embedding = nn.Parameter(torch.zeros(N_TARGET_FRAMES, units_embedding))
        self.dominant_hand_embedding = LandmarkEmbedding(units_embedding,N_COLUMNS)

    def forward(self, x):
        x = torch.where(
            x == 0.0,
            torch.tensor(0.0),
            (x - self.MEANS) / self.STDS,
        )
        x = self.dominant_hand_embedding(x)
        x = x + self.positional_embedding
        return x
class Net(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        n_target_frame,units_embedding,n_columns,means,stds, ## embedding
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
        classifier_class,
    ):

        super().__init__()

        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = Embedding(n_target_frame,units_embedding,n_columns,means,stds)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, classifier_class)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.transformer_encoder(x.float())
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x
    
if __name__ == '__main__' :
    model = Net(n_target_frame=cfg.n_target_frames,units_embedding=cfg.units_encoder,n_columns=cfg.n_columns,means=0.5,stds=0.5,
            d_model=cfg.units_encoder,nhead=cfg.nhead,dim_feedforward=cfg.dim_feedforward,num_layers=cfg.num_layers,
            dropout=cfg.dropout,classifier_class=cfg.classifier_class).to(cfg.device)