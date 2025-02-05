import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class MarketDataset(Dataset):
    def __init__(self, csv_file):
        """Load dataset and normalize features."""
        self.data = pd.read_csv(csv_file)
        
        # Select all feature columns except label and timestamp
        feature_columns = [col for col in self.data.columns if col not in ['label', 'timestamp', 'symbol']]
        self.features = self.data[feature_columns].values.astype(np.float32)
        self.labels = self.data['label'].values.astype(np.int64)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PatternClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=42):
        """Defines a simple fully connected neural network for pattern classification."""
        super(PatternClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    """Train the model and track loss/accuracy."""
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        model.train()
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    dataset = MarketDataset('data.csv')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = dataset.features.shape[1]
    model = PatternClassifier(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)
    torch.save(model.state_dict(), 'pattern_classifier.pth')
    print("Training complete and model saved.")