import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd()))
else:
    sys.path.append(os.path.join(os.getcwd(), '..', '..'))

from models.RsModel import RsModel
import torch, torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = RsModel().to(DEVICE)

batches_with_diff_length = [
    torch.randn(8,  2, 3, 250, 250).to(DEVICE), # seq len 2  batch size 8
    torch.randn(4,  6, 3, 250, 250).to(DEVICE), # seq len 6  batch size 4
    torch.randn(1, 11, 3, 250, 250).to(DEVICE)  # seq len 11 batch size 1
]

for batch in batches_with_diff_length:
    pred = model(batch)
    print(pred.shape)