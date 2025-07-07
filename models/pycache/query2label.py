# models/query2label.py
import torch
import torch.nn as nn
from torchvision import models

class Query2Label(nn.Module):
    def __init__(self, num_classes=29, backbone_name="resnet101", pretrained=True, embed_dim=512, num_queries=29):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # 1. Backbone (ResNet101 por padrão)
        if backbone_name == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
            layers = list(backbone.children())[:-2]  # Remove avgpool e fc
            self.backbone = nn.Sequential(*layers)
            self.backbone_out_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone_name} não implementado ainda.")

        # 2. Reduz dimensões para o transformer
        self.input_proj = nn.Conv2d(self.backbone_out_channels, embed_dim, kernel_size=1)

        # 3. Embeddings de rótulos
        self.label_embed = nn.Embedding(num_queries, embed_dim)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # 5. Camada de classificação (1 para cada label)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.backbone(x)                          # [B, 2048, H/32, W/32]
        features = self.input_proj(features)                # [B, E, H/32, W/32]
        B, C, H, W = features.shape
        features = features.flatten(2).permute(2, 0, 1)     # [HW, B, E]

        queries = self.label_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [N_CLASSES, B, E]

        out = self.transformer_decoder(tgt=queries, memory=features)   # [N_CLASSES, B, E]
        out = self.classifier(out).squeeze(-1).transpose(0, 1)         # [B, N_CLASSES]

        return out
