 # 使用PaddlePaddle复现论文《Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding》

# <a href="https://arxiv.org/pdf/1809.07426v1.pdf">Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding</a>
# 摘要：
Top-N顺序推荐模型将每个用户作为一个序列
，目的是预测前n名的排名
用户可能在“不久的将来”进行交互的项目。订单
交互作用意味着顺序模式起着重要作用
在此角色中，序列中较近期的项会产生较大的影响
下一个项目。在本文中，我们提出了一种卷积序列
嵌入推荐模型(Caser)作为解决方案
这个要求。这个想法是嵌入一系列最近的项目
转化成一个“意象”，在时间和潜在空间中循序渐进地学习
使用卷积滤波器将模式作为图像的局部特征。
该方法提供了一个统一和灵活的网络结构
捕获一般首选项和顺序模式。在公共数据集上的实验表明，Caser一贯如此
优于最先进的顺序推荐方法
在各种常见的评估指标上。

###　**环境要求**:
#### paddle>=2.0.0;scipy;numpy

## **项目结构**
#### caser_paddle
#### |__ datasets
#### $~~~~$|__ ml1m
#### $~~~~$|__ gowalla
#### |__model
#### $~~~~$|__ caser.py
#### |__ utils
#### $~~~~$|__ interactions.py
#### $~~~~$|__ utils.py
#### |__ checkpoint
#### |__run.py


### **目录结构及配置说明**：
#### **1.在datasets中存放训练和测试的数据集，本项目数据集是MovieLens-1m和Gowalla**
#### **2.model文件夹下为caser模型结构**
#### **3.utils文件夹下包含输入数据处理、模型验证指标等functions**
#### **4.checkpoint用于存放训练的模型参数**
#### **5.run.py 启动程序，包含定义的训练、预测功能。**
#### **说明**：**run.py 中重要参数如下**:
#### **在MovieLens-1m和Gowalla数据集上需设置以下不同的参数**
##### [--mode ] 选择训练或测试模式：默认为train，可设置test进行测试验证
##### [--train_root ] 训练集的路径
##### [--test_root] 测试集的路径
##### [--d] MovieLens-1m 中 d=50;Gowalla 中 d=100
##### [--nv] MovieLens-1m 中 nv=4;Gowalla 中 nv=2
##### [--ac_conv] MovieLens-1m 中 ac_conv='relu';Gowalla 中 ac_conv='iden'
##### [----ac_fc] MovieLens-1m 中 ac_fc='relu';Gowalla 中 ac_fc='sigm'

### **启动程序**：
### Usage:  python run.py --mode train --train_root Path --test_root Path
#### **说明** 根据数据集使用不同的超参数设置进行实验,重要参数如上参数说明。


## **复现结论**
#### **论文复现MAP指标要求：** 1、MovieLens MAP=0.1507 2、Gowalla MAP=0.0928
#### **本次论文复现精度如下：**
#### **1.MovieLens MAP=0.1752**
#### **2.Gowalla MAP=0.0947**
##### **【说明】训练30个epoch,取训练测试过程中最高的记录**

### **参考**:
#### 1、[https://github.com/graytowne/caser_pytorch/](http://)
#### 2、[https://github.com/graytowne/rank_distill/tree/master/datasets/gowalla](http://)
