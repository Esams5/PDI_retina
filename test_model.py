from models.query2label import Query2Label
import torch

model = Query2Label(num_classes=29, backbone_name="resnet101")
dummy_input = torch.randn(2, 3, 384, 384)  # batch de 2 imagens
output = model(dummy_input)

print("Shape da saída:", output.shape)  # Deve ser [2, 29]
