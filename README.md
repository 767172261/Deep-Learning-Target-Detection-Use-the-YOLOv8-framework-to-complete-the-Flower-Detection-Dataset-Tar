深度学习目标检测使用 YOLOv8 框架来完成花卉检测数据集目标检测任务 识别花卉的种类
@[toc]
以下文字及代码仅供参考。

花卉检测数据集5758张 15类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7108a5174e6b4612bde2c9b5479a2aab.png)


训练集4606张，验证集1152张，15类
分类：'daisy', 'dandelion', 'Rose', 'sunflower', 'champaka', 'chitrak', 'Common_Lanthana', 'Hibiscus', 'honeysuckle', 'indian_mallow', 'Jatropha', 'malabar_melastome', 'Marigold', 'shankupushpam', 'spider_lily'
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f5c1e3954dea4d029d0d3f439676a18e.png)

| 类别 | 图片数量 | 标注框数量 |
| ---- | ---- | ---- |
| daisy | 741 | 5070 |
| dandelion | 1079 | 2541 |
| Rose | 796 | 1589 |
| sunflower | 1102 | 3661 |
| champaka | 125 | 125 |
| chitrak | 174 | 174 |
| Common_Lanthana | 109 | 109 |
| Hibiscus | 97 | 97 |
| honeysuckle | 122 | 123 |
| indian_mallow | 227 | 227 |
| Jatropha | 356 | 521 |
| malabar_melastome | 241 | 242 |
| Marigold | 255 | 256 |
| shankupushpam | 237 | 239 |
| spider_lily | 117 | 117 |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99c481447e024fe7b9f69a9dc7ce054f.png)
 YOLO 数据集的花卉检测系统的完整实现方案，涵盖数据集划分、模型训练、评估以及推理的详细步骤。使用 YOLOv8 框架来完成目标检测任务，并支持通过 GUI 界面展示检测结果。

---

### **项目结构**

```
Flower_Detection/
├── data/                       # 数据集
│   ├── images/                 # 图片文件夹
│   ├── labels_txt/             # 标签文件(txt格式)
│   ├── train.txt               # 训练集列表
│   ├── val.txt                 # 验证集列表
├── models/                     # 模型权重
│   ├── yolov8.pt               # YOLOv8预训练权重
│   ├── best.pt                 # 训练后的最佳权重
├── utils/                      # 工具函数
│   ├── dataset_utils.py        # 数据集处理工具
│   ├── eval_utils.py           # 评估工具
├── gui/                        # GUI 文件
│   ├── main_gui.py             # 主界面逻辑
│   ├── ui_main.ui              # PyQt5 UI 设计文件
├── requirements.txt            # Python 依赖包列表
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── inference.py                # 推理脚本
└── README.md                   # 项目说明文档
```

---

### **1. 安装依赖**

在 `requirements.txt` 中列出所需的依赖库：

```txt
torch==2.0.1
torchvision==0.15.2
opencv-python==4.7.0.72
PyQt5==5.15.9
ultralytics==8.0.43          # YOLOv8 库
numpy==1.23.5
matplotlib==3.7.1
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

### **2. 数据集划分**

将 5758 张图片按照 8:2 的比例划分为训练集和验证集。

```python
import os
import random

def split_dataset(image_dir, label_dir, train_ratio=0.8):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    train_split = int(len(image_files) * train_ratio)
    train_files = image_files[:train_split]
    val_files = image_files[train_split:]

    with open("data/train.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("data/val.txt", "w") as f:
        f.write("\n".join(val_files))
```

运行数据集划分：

```python
split_dataset("data/images", "data/labels_txt")
```

---

### **3. 配置 YOLOv8**

创建 `yolov8.yaml` 文件，定义类别和数据集路径。

```yaml
train: data/train.txt
val: data/val.txt

nc: 15  # 类别数
names: ['daisy', 'dandelion', 'Rose', 'sunflower', 'champaka', 'chitrak', 'Common_Lanthana', 
        'Hibiscus', 'honeysuckle', 'indian_mallow', 'Jatropha', 'malabar_melastome', 
        'Marigold', 'shankupushpam', 'spider_lily']
```

---

### **4. 训练与评估**

#### 训练脚本 (`train.py`)

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("models/yolov8.pt")

# 开始训练
model.train(data="data/yolov8.yaml", epochs=100, imgsz=640, batch=16, device=0)
```

#### 评估脚本 (`evaluate.py`)

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/train/weights/best.pt")

# 在验证集上评估
metrics = model.val()
print(metrics)
```

---

### **5. 构建 GUI 应用程序**

#### `main_gui.py`

```python
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap
from ui_main import Ui_MainWindow
from inference import run_inference
import cv2

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # 连接按钮事件
        self.btn_image.clicked.connect(self.select_image)
        self.btn_video.clicked.connect(self.select_video)
        self.btn_camera.clicked.connect(self.start_camera)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            result_image = run_inference(file_path)
            self.display_image(result_image)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            cap = cv2.VideoCapture(file_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = run_inference(frame)
                self.display_image(result_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = run_inference(frame)
            self.display_image(result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_image.setPixmap(QPixmap.fromImage(q_img))
```

---

### **6. 推理脚本 (`inference.py`)**

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("runs/train/weights/best.pt")

def run_inference(input_data):
    if isinstance(input_data, str):  # 如果是图片路径
        results = model(input_data)
        img = cv2.imread(input_data)
    else:  # 如果是视频帧
        results = model(input_data)
        img = input_data

    # 绘制检测框
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img
```

---

### **7. 运行程序**

```bash
python main_gui.py
```

---

### **8. 总结**

基于 YOLOv8 的花卉检测系统，支持图片、视频、摄像头输入，并通过 PyQt5 界面展示检测结果。系统能够检测 15 类花卉目标，并提供直观的可视化效果。

以上仅供参考。


