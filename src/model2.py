from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集加载
dataset = datasets.ImageFolder("D:\\train", transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# 模型定义
model = models.resnet18(weights='DEFAULT').to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练过程
num_epochs = 2
best_acc = 0.0
save_dir = 'D:\\final_model1'
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 添加 tqdm 进度条
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

def evaluate(model, criterion, dataloader, device):
    model.eval()  # 修正此处的错误，添加缺失的调用
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # 添加 tqdm 进度条
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    val_loss, val_acc = evaluate(model, criterion, val_loader, device)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    scheduler.step()

# 测试集评估
model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth')))
test_loss, test_acc = evaluate(model, criterion, test_loader, device)
print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
