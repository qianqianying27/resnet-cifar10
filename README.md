# resnet-cifar10
ResNet-18 CIFAR-10 图像分类
使用 PyTorch 从零复现 ResNet-18，在 CIFAR-10 数据集上完成图像分类任务。
项目背景
本项目是算法工程师-机器学习"的技能实践，重点训练：
深度学习模型从零实现能力
数据增强与训练优化技巧
模型训练的可视化与工程化
环境要求
Python 3.8+
PyTorch 1.12+
torchvision
tensorboard
tqdm
matplotlib
bash
pip install torch torchvision tensorboard tqdm matplotlib
快速开始
bash
git clone https://github.com/qianqianying27/resnet-cifar10.git
cd resnet-cifar10
python train.py
项目结构
plain
resnet-cifar10/
├── train.py          # 主训练脚本
├── model.py          # 手写 ResNet-18 模型定义
├── utils.py          # 训练/测试辅助函数
├── config.py         # 超参数配置
├── data/             # 数据集自动下载
├── checkpoints/      # 模型保存
├── runs/             # TensorBoard 日志
└── results/
    ├── accuracy_plot.png
    └── loss_plot.png
ResNet-18 模型结构
plain
输入: [B, 3, 32, 32]
  ↓
Conv3x3 → BN → ReLU
  ↓
Stage1: 2个 BasicBlock, 通道64
  ↓
Stage2: 2个 BasicBlock, 通道128, stride=2
  ↓
Stage3: 2个 BasicBlock, 通道256, stride=2
  ↓
Stage4: 2个 BasicBlock, 通道512, stride=2
  ↓
Global Average Pooling
  ↓
FC: 512 → 10
  ↓
输出: [B, 10]
核心：残差连接 (Residual Connection)
Python
# BasicBlock 前向传播
identity = x                    # 保存原始输入
out = conv1(x) → bn1 → relu
out = conv2(out) → bn2
out += identity                 # 残差连接
out = relu(out)
残差连接解决了深层网络的梯度消失问题，使得训练超过100层的网络成为可能。
关键技术点
1. 数据增强
Python
transforms.RandomCrop(32, padding=4)      # 随机裁剪，四周填充4像素
transforms.RandomHorizontalFlip(p=0.5)  # 50%概率水平翻转
transforms.Normalize(mean, std)           # 标准化
2. 训练配置
表格
参数	值	说明
优化器	SGD + Momentum(0.9)	ResNet论文标准配置
初始学习率	0.1	配合StepLR衰减
学习率调度	MultiStepLR	第50、75epoch衰减为0.1倍
权重衰减	5e-4	L2正则化，防止过拟合
Batch Size	128	根据显存调整
Epochs	100	完整收敛
3. 与简单CNN基线对比
表格
模型	参数量	测试准确率
简单CNN (3层)	~120K	~72%
ResNet-18	~11M	~92%
实验结果
训练曲线
运行以下命令查看：
bash
tensorboard --logdir=runs
浏览器打开 http://localhost:6006
关键指标
最佳测试准确率: 92.3% (Epoch 89)
训练时间: ~45分钟 (RTX 3060)
收敛epoch: ~60
代码详解
model.py
手写 ResNet-18，包含：
BasicBlock: 基础残差块
ResNet: 主网络
resnet18(): 工厂函数
utils.py
train_epoch(): 训练一个epoch
test_epoch(): 测试一个epoch
save_checkpoint(): 保存模型
plot_history(): 绘制训练曲线
config.py
所有超参数集中管理，方便调参实验。
参考
He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
PyTorch CIFAR-10 Tutorial
ResNet Paper
后续计划
[ ] 尝试 ResNet-34 更深网络
[ ] 加入 Cutout / Mixup 数据增强
[ ] 尝试 CosineAnnealingWarmRestarts 学习率调度
[ ] 模型量化与 ONNX 导出
