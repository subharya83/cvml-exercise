import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from SubCentShotClassification import SGNet
from torch.nn import CrossEntropyLoss

# Assuming MovieShotsDataset is a custom dataset class
from datasets import MovieShotsDataset

# Hyperparameters
num_epochs = 60
batch_size = 128
learning_rate = 0.001
momentum = 0.9

def train_model():
    # Initialize dataset and dataloader
    train_dataset = MovieShotsDataset(root='path_to_train_data', split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SGNet()
    criterion_scale = CrossEntropyLoss()
    criterion_movement = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for i, (images, scale_labels, movement_labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            scale_logits, movement_logits = model(images)

            # Compute loss
            loss_scale = criterion_scale(scale_logits, scale_labels)
            loss_movement = criterion_movement(movement_logits, movement_labels)
            total_loss = loss_scale + loss_movement

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}')

        # Adjust learning rate
        if epoch == 20 or epoch == 40:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

    # Save the model
    torch.save(model.state_dict(), 'sgnet_model.pth')

if __name__ == "__main__":
    train_model()