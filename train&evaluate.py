#training function
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


def get_model_name(name, batch_size, learning_rate, epoch):
    return f"model_{name}_bs{batch_size}_lr{learning_rate}_epoch{epoch}"

def train_net(net, train_loader, val_loader, y_mean, y_std, batch_size=64, learning_rate=1e-3, num_epochs=50, checkpoint_freq=5):
    device = torch.device("cpu")
    net.to(device)

    criterion = nn.L1Loss()  # MAE loss on normalized values
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    print("Starting training...\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0.0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        train_loss[epoch] = total_train_loss / total_samples

        # -------- Validation & Prediction --------
        net.eval()
        val_outputs = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.float().view(-1, 1).to(device)

                outputs = net(inputs)

                val_outputs.append(outputs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        # Stack and flatten
        pred_norm = np.concatenate(val_outputs).flatten()
        true_norm = np.concatenate(val_labels).flatten()

        # Inverse Z-score to get real ages
        pred_age = pred_norm * y_std + y_mean
        true_age = true_norm * y_std + y_mean

        # MAE in real age units
        real_mae = np.mean(np.abs(pred_age - true_age))
        val_loss[epoch] = real_mae

        # Print info
        print(f"Epoch {epoch+1}: Train Loss (norm) = {train_loss[epoch]:.4f} | Val MAE (age) = {real_mae:.2f} years")
        print("Sample predictions:")
        indices = random.sample(range(len(pred_age)), 5)
        for i in indices:
          print(f"  Predicted: {pred_age[i]:.1f}  |  Actual: {true_age[i]:.1f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
            torch.save(net.state_dict(), f"model_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved: model_epoch_{epoch+1}.pt\n")

    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds.")
    return train_loss, val_loss

#evaluation function
def evaluate(net, loader, criterion, device):
    net.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device).float().view(-1, 1)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss

#helper function
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path
