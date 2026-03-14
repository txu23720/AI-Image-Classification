import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# 设置设备为 CUDA（如果可用）或 CPU  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据增强变换，与训练时保持一致  
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小  
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化  
])

# 加载预训练的 ResNet18 模型（但这里不使用预训练权重），并修改最后一层以适应我们的二分类任务  
model_path = "D:\\final_model1\\model.pth"
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 设置为二分类
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file {model_path} not found.")
    exit()
model = model.to(device)
model.eval()  # 设置为评估模式

# 定义预测函数
def predict_image(image_path):
    """
    预测给定图像路径的图像类别。

    参数:
    image_path (str): 图像文件的路径。

    返回:
    int: 预测类别（0 表示原始图像，1 表示 AI 生成图像），如果处理失败则返回 None。
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()  # 返回预测类别的索引
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# 预测一组图像并保存结果到 CSV 文件
image_folder = 'D:\\testset'  # 图像文件夹路径
image_files = sorted(os.listdir(image_folder), key=lambda x: x.lower())  # 按字典序排序图像文件
predictions = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    prediction = predict_image(image_path)
    if prediction is not None:
        predictions.append({'Image': os.path.splitext(image_file)[0], 'Prediction': prediction})  # 0 表示原始图像，1 表示 AI 生成图像

# 将预测结果保存为 CSV 文件，不包含表头
df = pd.DataFrame(predictions)
csv_path = 'D:\\cla_pre.csv'
df.to_csv(csv_path, index=False, header=False)  # 不保存索引和表头
print(f'Predictions saved to {csv_path}')
