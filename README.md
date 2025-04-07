# Pinyin2Hanzi_Transformer

基于`Transformer`的预训练汉语拼音序列转汉字序列模型，训练数据全部采用`Duyu/Pinyin-Hanzi`（[单击此处](https://huggingface.co/datasets/Duyu/Pinyin-Hanzi) 跳转至数据集页面）

## 📖 模型概览

| 模型相关参数 | 数值 |
| ----- | ----- |
| 参数量 | 62,200,184 (62M) |
| 可学习参数量比例 | 100% |
| Transformer编码器层数 | 8 |
| Transformer解码器层数 | 6 |
| 词嵌入维度 | 512 |
| 多头注意力层注意力头数 | 16 |
| Transformer前馈层维度 | 1024 |
| 截断长度 | 14 |

## 🚀 快速启动

### 硬件要求
**模型推理：** 轻量级模型，常见的家用计算机配置（`8GB`内存及以上）即可运行，若有NVIDIA GPU（使用`CUDA`）可加速。

**模型训练：** 可在`CPU`设备上运行（`8GB`内存及以上），但训练速度非常慢，建议使用`NVIDIA GeForce RTX 2080`及以上配置的加速卡GPU。

### 使用方法

**（一）环境准备：**
1. 硬件要求：如上所述
2. 依赖安装：使用的第三方库包括`numpy`、`pandas`、`torch`

```bash
pip install numpy pandas
# 请查看PyTorch官方文档，以进行torch的安装。
```

3. 下载代码（`run.py`）及预训练权重（`pinyin2hanzi_transformer.pth`）

- 下载地址：`Hugging Face`平台: https://huggingface.co/Duyu/Pinyin2Hanzi-Transformer 或`GitHub`平台: https://github.com/duyu09/Pinyin2Hanzi-Transformer/releases/tag/Pinyin2Hanzi-Transformer-v1.0

**（二）模型推理：** 
1. 解除主函数中`use_main()`的注释，增加`train_main()`的注释。
2. 修改`use_main()`中的模型文件路径及汉语拼音序列。
3. 运行代码，实现预测。

**（三）模型训练：** 
1. 准备好适当的训练环境（带加速显卡的机器）
2. 准备数据集
  - 文件格式：`CSV`文件。
  - 第一列是汉字序列。
  - 第二列是拼音序列，每个汉字对应的拼音用一个空格隔开。

3. 解除主函数中`train_main()`的注释，增加`use_main()`的注释。
4. 根据情况，修改`train_main()`中的各项参数。
5. 运行代码，开始训练。

## 🎓 项目作者

**DuYu** (Chinese Simplified: **杜宇**, No.202103180009, qluduyu09@163.com), Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences).

## 📊 访客统计

项目在Hugging Face平台同步开源：https://huggingface.co/Duyu/Pinyin2Hanzi-Transformer

<div><b>Number of Total Visits (All of Duyu09's GitHub Projects): </b><br><img src="https://profile-counter.glitch.me/duyu09/count.svg" /></div> 

<div><b>Number of Total Visits (Pinyin2Hanzi-Transformer): </b>
<br><img src="https://profile-counter.glitch.me/duyu09-Pinyin2Hanzi-Transformer/count.svg" /></div>


