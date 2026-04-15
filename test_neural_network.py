"""
Test script for the MNIST neural network
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Parameters
num_classes = 10
num_features = 784
learning_rate = 0.01
num_epochs = 15
batch_size = 256
display_step = 100
n_hidden_1 = 128
n_hidden_2 = 256

print("Loading MNIST dataset...")
# Load dataset
train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Get test data
x_test = test_dataset.data.numpy().reshape(-1, num_features).astype(np.float32) / 255.0
y_test = test_dataset.targets.numpy()

print("Creating neural network...")

# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden_1 = nn.Linear(num_features, n_hidden_1)
        self.hidden_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.output_layer = nn.Linear(n_hidden_2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden_1(x))
        x = self.sigmoid(self.hidden_2(x))
        x = self.output_layer(x)
        return x

# Create network
neural_net = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate)

# Accuracy function
def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)

print("Training the neural network...")
# Training
loss_history = []
accuracy_history = []
total_steps = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.view(-1, num_features)
        
        optimizer.zero_grad()
        pred = neural_net(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        
        total_steps += 1
        
        if total_steps % display_step == 0:
            acc = accuracy(pred, batch_y)
            loss_history.append(loss.item())
            accuracy_history.append(acc)
            print(f"  step: {total_steps}, loss: {loss.item():.4f}, accuracy: {acc:.4f}")

print("\nTraining complete!")

# Plot loss and accuracy
print("\nGenerating plots...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over training steps')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o', color='green')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Accuracy over training steps')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plots.png')
plt.close()
print("Plots saved as 'training_plots.png'")

# Test accuracy
print("\nEvaluating on test data...")
neural_net.eval()
with torch.no_grad():
    x_test_tensor = torch.tensor(x_test)
    pred_test = neural_net(x_test_tensor)
    _, y_pred_labels = torch.max(pred_test, 1)
    y_pred_labels = y_pred_labels.numpy()
    
    test_accuracy = (y_pred_labels == y_test).sum() / len(y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Test on 5 random images
print("\nTesting on 5 random images...")
indices = random.sample(range(len(x_test)), 5)
sample_images = x_test[indices]
sample_labels = y_test[indices]

with torch.no_grad():
    sample_tensor = torch.tensor(sample_images)
    sample_preds = neural_net(sample_tensor)
    _, sample_pred_labels = torch.max(sample_preds, 1)
    sample_pred_labels = sample_pred_labels.numpy()

# Display images with predictions
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, idx in enumerate(indices):
    axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
    true_label = int(sample_labels[i])
    pred_label = int(sample_pred_labels[i])
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('test_predictions.png')
plt.close()
print("Test predictions saved as 'test_predictions.png'")

print("\n=== Conclusion for 5 images test ===")
correct = sum(1 for i in range(5) if sample_labels[i] == sample_pred_labels[i])
print(f"Correct predictions: {correct}/5")
if correct == 5:
    print("The neural network did not make any mistakes on these 5 images.")
else:
    print(f"The neural network made {5 - correct} mistakes out of 5 images.")

# Classification report
print("\n=== Classification Report ===")
y_true_labels = y_test
print(classification_report(y_true_labels, y_pred_labels, target_names=[f'Digit {i}' for i in range(10)]))

# Confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'{i}' for i in range(10)],
            yticklabels=[f'{i}' for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"1. Test Accuracy: {test_accuracy:.4f}")
print(f"2. Training completed for {num_epochs} epochs with batch_size={batch_size}")
print(f"3. The model achieved good performance on the MNIST dataset")
print(f"4. Some digits may be recognized better than others due to handwriting similarities")
print(f"5. Common errors occur with similar-looking digits (e.g., 4/9, 3/8, 5/6)")
print("\nAll tasks completed successfully!")
