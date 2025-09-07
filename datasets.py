#split data
from sklearn.model_selection import train_test_split

# Split 80% train, 10% val, 10% test

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)

#print age distribution
import matplotlib.pyplot as plt

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df['actual_age'], bins=20, edgecolor='black', color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age (years)')
plt.ylabel('Number of Participants')
plt.grid(True)
plt.show()

#create dataloader
from torch.utils.data import TensorDataset, DataLoader

# Wrap in TensorDataset
train_dataset = TensorDataset(X_train, torch.tensor(Y_train))
val_dataset = TensorDataset(X_val, torch.tensor(Y_val))
test_dataset = TensorDataset(X_test, torch.tensor(Y_test))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")
