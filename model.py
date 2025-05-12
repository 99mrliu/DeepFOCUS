import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class Inception1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))
        b4 = F.relu(self.branch4(x))
        return torch.cat([b1, b2, b3, b4], dim=1)


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mid_channels = channels // reduction
        self.conv_seq = nn.Sequential(
            nn.Conv1d(1, self.mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.mid_channels, channels, 1),
            nn.Sigmoid()
        )
        self.conv_feat = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.mid_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T = x.size()
        x_seq = x.mean(1, keepdim=True)
        x_seq = self.conv_seq(x_seq)
        x_feat = x.mean(2, keepdim=True)
        x_feat = self.conv_feat(x_feat)
        x_feat = x_feat.expand(-1, -1, T)
        attention = torch.sigmoid(x_seq + x_feat)
        return x * attention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = CoordinateAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_out = self.attention(x.permute(0, 2, 1))
        attn_out = attn_out.permute(0, 2, 1)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.permute(0, 2, 1)


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 9
        self.inception = nn.Sequential(
            Inception1D(self.in_channels),
            Inception1D(128)
        )
        self.transformer = nn.Sequential(
            TransformerBlock(128),
            TransformerBlock(128)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.inception(x)
        x = self.transformer(x)
        return self.classifier(x)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target.long())
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        early_stopping(val_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            preds.extend(output.squeeze().round().cpu().numpy())
            labels.extend(target.cpu().numpy())

    accuracy = np.mean(np.equal(preds, labels))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    model = CustomModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

