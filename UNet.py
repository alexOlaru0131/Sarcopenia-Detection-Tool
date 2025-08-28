import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vertebraDataset import VertebraDataset

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(1, 16)
        self.enc2 = CBR(16, 32)
        self.enc3 = CBR(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = CBR(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = CBR(64, 32)

        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec0 = CBR(32, 16)

        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        u2 = self.up2(b)
        u2 = self._pad_to_match(u2, e3)
        d2 = self.dec2(torch.cat([u2, e3], dim=1))

        u1 = self.up1(d2)
        u1 = self._pad_to_match(u1, e2)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u0 = self.up0(d1)
        u0 = self._pad_to_match(u0, e1)
        d0 = self.dec0(torch.cat([u0, e1], dim=1))

        return torch.sigmoid(self.out(d0))

    def _pad_to_match(self, src, target):
        diffY = target.shape[2] - src.shape[2]
        diffX = target.shape[3] - src.shape[3]
        return F.pad(src, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])

def dice_loss(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + dice_loss(pred, target)


def train():
    device = torch.device("cuda")

    dataset = VertebraDataset("dataset/images", "dataset/masks")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ComboLoss()

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()

        print(f"Epoca {epoch+1} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "vertebra_unet.pth")
    print("Model salvat: vertebra_unet.pth")

if __name__ == "__main__":
    train()