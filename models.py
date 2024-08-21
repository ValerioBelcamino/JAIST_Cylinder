import torch.nn as nn
from ViViT.vivit import ViViT
import torch 
import math 

class HAR_Transformer(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim, max_seq_length):
        super(HAR_Transformer, self).__init__()
        self.input_dim = input_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length
        
        self.input_embedding = nn.Linear(input_dim, dim_feedforward)
        self.position_encoding = nn.Parameter(self.create_position_encoding(max_seq_length, dim_feedforward), requires_grad=False)
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward,
                                                     batch_first=True, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dp = nn.Dropout(0.3)
        self.fc = nn.Linear(dim_feedforward, output_dim)

    def create_position_encoding(self, max_seq_length, dim_feedforward):
        position_encoding = torch.zeros(max_seq_length, dim_feedforward)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_feedforward, 2).float() * -(math.log(10000.0) / dim_feedforward))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)  # Shape: (1, max_seq_length, dim_feedforward)
        return position_encoding

    def forward(self, x, lengths):
        x = self.input_embedding(x)
        x = x + self.position_encoding[:, :x.size(1), :]

        # Create attention masks
        mask = self.create_attention_mask(lengths, x.size(1), device=x.device)
        # x = x.permute(1, 0, 2)  # (batch_size, seq_length, dim) -> (seq_length, batch_size, dim)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        # x = x.permute(1, 0, 2)  # (seq_length, batch_size, dim) -> (batch_size, seq_length, dim)
        x = x.mean(dim=1)  # Average pooling over the sequence length
        x = self.dp(x)
        x = self.fc(x)
        return x

    def create_attention_mask(self, lengths, max_length, device):
        mask = torch.arange(max_length, device=device).expand(len(lengths), max_length) >= lengths.unsqueeze(1)
        return mask

    
class CHARberoViVit(nn.Module):
    def __init__(self, pixel_dim, patch_size, n_classes, max_seq_len, n_features, nhead, num_encoder_layers, dim_feedforward, intermediate_dim):
        super(CHARberoViVit, self).__init__()
        self.ViViT_branch1 = ViViT(pixel_dim, patch_size, intermediate_dim, max_seq_len, in_channels=1)
        self.ViViT_branch2 = ViViT(pixel_dim, patch_size, intermediate_dim, max_seq_len, in_channels=1)
        self.HAR_Transformer_branch = HAR_Transformer(n_features, nhead, num_encoder_layers, dim_feedforward, intermediate_dim, max_seq_len)
        self.fc = nn.Linear(3 * intermediate_dim, n_classes)

    def forward(self, x1, x2, x3, lengths):
        x1 = self.ViViT_branch1(x1)
        x2 = self.ViViT_branch2(x2)
        x3 = self.HAR_Transformer_branch(x3, lengths)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x


class BicefHARlo(nn.Module):
    def __init__(self, pixel_dim, patch_size, n_classes, max_seq_len, intermediate_dim):
        super(BicefHARlo, self).__init__()
        self.ViViT_branch1 = ViViT(pixel_dim, patch_size, intermediate_dim, max_seq_len, in_channels=1)
        self.ViViT_branch2 = ViViT(pixel_dim, patch_size, intermediate_dim, max_seq_len, in_channels=1)
        self.fc = nn.Linear(2 * intermediate_dim, n_classes)

    def forward(self, x1, x2):
        x1 = self.ViViT_branch1(x1)
        x2 = self.ViViT_branch2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x