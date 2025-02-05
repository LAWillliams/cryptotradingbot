import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer import MarketDataset, train_model
from trainer import DataLoader
from pattern import PatternClassifier

def main():
    # Create the dataset. Ensure 'data.csv' is created from your preprocessing step.
    dataset = MarketDataset('data.csv')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Determine input dimensions from the dataset
    input_dim = dataset.features.shape[1]
    hidden_dim = 64        # You can tune this hyperparameter
    output_dim = 42        # Assuming 42 pattern classes
    
    model = PatternClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)
    
    # Save the trained model for later use in pattern detection
    torch.save(model.state_dict(), 'pattern_classifier.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()
