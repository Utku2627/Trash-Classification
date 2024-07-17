import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from PIL import Image

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"Torch is using: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not usable.")

torch.cuda.empty_cache()


# HYPERPARAMETERS
IMG_SIZE = (512, 384)
batch_size = 32
learning_rate = 2e-5
num_epochs = 20


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Load data
def load_custom_data(data_dir):
    data = []
    labels = []
    categories = os.listdir(data_dir)
    label_map = {category: index for index, category in enumerate(categories)}

    for category in categories:
        category_dir = os.path.join(data_dir, category)
        for image_file in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            image = image.resize(IMG_SIZE)
            image_np = np.array(image)
            data.append(image_np)
            labels.append(label_map[category])

    return data, labels


# Split data
def prepare_train_valid_test(base_dir, train_folder, valid_folder, test_folder):
    train_data, train_labels = load_custom_data(os.path.join(base_dir, train_folder))
    valid_data, valid_labels = load_custom_data(os.path.join(base_dir, valid_folder))
    test_data, test_labels = load_custom_data(os.path.join(base_dir, test_folder))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(train_data, train_labels, transform=transform)
    valid_dataset = CustomDataset(valid_data, valid_labels, transform=transform)
    test_dataset = CustomDataset(test_data, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# AlexNet Model
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.5)

        dummy_input = torch.randn(1, 3, 512, 384)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def _forward_conv(self, x):
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = self.pool3(nn.ReLU()(self.conv5(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.dropout(nn.ReLU()(self.fc2(x)))
        x = self.fc3(x)
        return x


# Training and Evaluation Functions
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            if isinstance(preds, torch.Tensor) and isinstance(labels, torch.Tensor):
                correct_preds += (preds == labels).sum().item()
            else:
                raise TypeError("preds and labels should be of type torch.Tensor")
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct_preds / total_preds

    return epoch_loss, accuracy, all_labels, all_preds


# Main Function
def main():
    base_dir = r'C:\TrashClassification\splitted_data'
    train_folder = 'train'
    valid_folder = 'validation'
    test_folder = 'test'

    train_loader, valid_loader, test_loader = prepare_train_valid_test(base_dir, train_folder, valid_folder,
                                                                       test_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=len(os.listdir(os.path.join(base_dir, train_folder)))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_accuracy, _, _ = evaluate_model(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, "
            f"Validation Accuracy: {valid_accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()

    test_loss, test_accuracy, all_labels, all_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Define the labels
    labels = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix (Accuracy: {test_accuracy:.4f})')
    plt.show()


if __name__ == "__main__":
    main()
