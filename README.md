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
