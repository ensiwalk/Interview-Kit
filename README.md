# Interview-Kit
2022秋招的临阵磨枪

## 机器学习篇

### 1. Batch Normalization

#### 基本公式

$$
\mu = \frac{1}{m}\Sigma_{i=1}^mx_i\\
\sigma^2 = \frac{\Sigma_{i=1}^m(x_i-\mu)^2}{m}\\
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\\
y_i = \gamma\hat{x}_i+\beta
$$

$\epsilon$ 是为了防止方差为0，$\gamma$ 和 $\beta$ 是可学习参数，为了使BN后的数据仍保留一定的原有特征，因为二者选择的比较好，可以使处理后的数据回归原始数据。

```python
def Batchnorm_simple_for_train(x, gamma, beta, bn_param):
"""
param:x    : 输入数据，设shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
	eps      : 接近0的数，防止分母出现0
	momentum : 动量参数，一般为0.9， 0.99， 0.999
	running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
	running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
	running_mean = bn_param['running_mean']  #shape = [B]
    running_var = bn_param['running_var']    #shape = [B]
	results = 0. # 建立一个新的变量
    
	x_mean=x.mean(axis=0)  # 计算x的均值
    x_var=x.var(axis=0)    # 计算方差
    x_normalized=(x-x_mean)/np.sqrt(x_var+eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移

    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var
    
    #记录新的值
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var 
    
	return results , bn_param

def Batchnorm_simple_for_test(x, gamma, beta, bn_param):
"""
param:x    : 输入数据，设shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
	eps      : 接近0的数，防止分母出现0
	momentum : 动量参数，一般为0.9， 0.99， 0.999
	running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
	running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
	running_mean = bn_param['running_mean']  #shape = [B]
    running_var = bn_param['running_var']    #shape = [B]
	results = 0. # 建立一个新的变量
   
    x_normalized=(x-running_mean )/np.sqrt(running_var +eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移
    
	return results , bn_param
```



### 2. 梯度消失和梯度爆炸问题

#### 现象描述

梯度消失时，接近于输出层的参数有所更新，而远离输出层，靠近输入层的参数则几乎不变；

梯度爆炸时，一步走的太远，性能变化飘忽不定（猜测）

#### 原因分析

总体原因是网络的反向传播算法需要不断地进行相乘。

* 梯度消失：网络层数过深，小梯度越乘越小；用了不合适的激活函数，比如sigmoid函数（sigmoid的梯度不可能超过0.5）
* 梯度爆炸：网络层数；初始化参数过大

#### 解决方案

* 预训练加微调
* 梯度剪切，防止梯度爆炸
* 换损失函数，用relu及其变体
* BN

$$
f_2= f_1(w^Tx+b)\\
\frac{\part f_2}{\part w} = \frac{\part f_2}{\part f_1}x
$$

可以看到，求梯度的时候有一项与输入$x$有关，而BN消除了$x$ 缩放带来的影响

* ResNet，因为有跨层连接，所以梯度有直通管道，可以增加深度

### 3. 监督问题

* 监督：有标签

* 无监督：无标签，聚类，降维问题

* 半监督：少量有标签，大量无标签

  伪标签法，用少量标签先训一个模型，然后用该模型去预测无标注的数据标签，构成新的数据集，然后再去训练，以此类推；

  EM算法，元学习

* 自监督：输入就是标签，自编码器

### 4. domain adaption

迁移学习/元学习，样本迁移，特征迁移，模型迁移

### 5. 样本不平衡如何处理

* 欠采样，去除一些多数样本
* 过采样，增加小样本的数量；数据增强，增加少数样本，mix-up，旋转，反向，加噪，相位等等
* 加权重，给小样本赋予更多的权重，penalized-SVM
* 集成学习：通过训练多个模型的方式解决数据不均衡的问题，是指将多数类数据随机分成少数类数据的量N份，每一份与全部的少数类数据一起训练成为一个分类器，这样反复训练会生成很多的分类器
* 特征选择，选取好区分的特征

### 6. KNN相关

#### K近邻算法

k近邻算法输入的是样本的特征向量，对应特征空间的点，输出是样本所属的类别。k近邻实际上是利用训练数据对特征向量空间进行划分，并将其“分类”的模型；

k近邻模型的三要素是`k值的选取`,`距离的度量`,`分类决策规则`；

具体而言，对于一个输入样本，选取距离样本最近的k个点，利用这k个点进行类别投票，选择票数最多的类别作为输入样本的类别。

##### 距离度量

$P(x_1,y_1),Q(x_2,y_2)$为例

* 曼哈顿距离：坐标差的绝对值之和

$$
dis(P,Q) = |x_1-x_2| + |y_1-y_2|
$$

* 欧式距离：平方开根号

$$
dis(P,Q) = \sqrt{(x_1-x_2)^2+(y_1-y_2)^2}
$$

* Lp距离
  $$
  dis(P,Q) = [(x_1-x_2)^p+(y_1-y_2)^p]^{1/p}
  $$
  
* 汉明距离：两个等长字符串中不同位置的数目

$$
\begin{aligned}
x = x_1,x_2,x_3,\cdots,x_n\\
y = y_1,y_2,y_3,\cdots,y_n\\
dis(x,y) = \Sigma_{i=1}^n\delta(x_i-y_i)
\end{aligned}
$$

* 余弦距离：余弦相似度是通过测量两个向量夹角的度数来度量他们之间的相似度。0度的相似度是1，90度的相似度是0，180的相似度是-1，不是真正的距离，不满足三角不等式。

$$
dis(x,y) = \frac{x^Ty}{\sqrt{||x||_2^2||y||_2^2}}
$$

* 马氏距离：马氏距离是对特征按照主成分进行旋转，让维度间相互独立，然后进行标准化，让维度同分布之后的欧氏距离。这样可以消除量纲，特征分布分散的影响；由于多维空间中，不同的特征之间可能是相关的，所以单独在各个维度上消除量纲也不行

$$
\begin{aligned}
dis(x,y) = \sqrt{(x-y)^TS^{-1}(x-y)}\\
S = \frac{1}{n}\Sigma(x-\mu)(y-\mu)^T
\end{aligned}
$$

##### k值的选择

选取较小的k值，代表模型复杂，因此近似误差会比较小，但是估计误差会增大，容易发生过拟合，被噪声影响

选择较大的k值，代表模型简单，近似误差较大，估计误差较小，容易欠拟合，当k取最大N时，则代表每次的预测结果都是数量最多的类别。

##### 分类决策规则

多数表决规则相当于损失函数为01函数时的经验风险最小化

#### kd树模型

这里假设k为维度，很简单，对一批数据有$(x_1,\cdots,x_k)$​ 共k个维度的特征，首先对第一个维度，取其中位数，左子树为所有在第一个维度上小于中位数的样本，右子树为所有在第一个特征维度上大于中位数的样本，依次类推，对k个维度划分下去，这样便得到了一棵深度为k+1的二叉树。先后顺序上选择方差更大的维度在前。

当来了一个新样本时，首先找到叶子节点作为“当前最近点”，以新样本为球心，以距离“当前最近点”为半径画球，如果与同级另一颗子树相交，则有更近点，更新“当前最近点”集合，一路递归上去到根节点，就找到了最终的最近点。

### 7. LR推导

#### 最大似然估计

最大似然估计的基本思想：拥有一组数据的样本观测值，并且已知其含参数概率分布的形式，选取合适的参数估计值，使得样本取到样本值的概率最大。

* 样本

设$(X_1,X_2,\cdots,X_n)$是来自总体的一个容量为n的样本，$(x_1,x_2,\cdots,x_n)$ 是相应的样本值。

* 离散型总体的似然函数

设分布律为$P(X=x) = p(x;\theta),\theta \in \Theta$的形式已知，$\theta$为待估参数，则样本的观察值为$(x_1,x_2,\cdots,x_n)$ 的概率为
$$
P(X_1=x_1,X_2=x_2,\cdots,X_n=x_n) = \Pi_{i=1}^nP(X_i=x_i)
$$
似然函数为
$$
L(\theta) = \Pi_{i=1}^np(x_i;\theta)
$$
对数似然函数为
$$
lnL(\theta) = \Sigma_{i=1}^nlnp(x_i;\theta)
$$

* 连续型总体的似然函数

设$Y \sim f(x;\theta)$​，$\theta$​ 为待估参数，则上述样本的联合概率密度为
$$
L(\theta) = L(x_1,\cdots,x_n;\theta) = \Pi_{i=1}^nf(x_i;\theta)
$$
因为样本在样本点附近取值为大概率事件，所以要最大化上述似然函数

对数似然函数为
$$
lnL(\theta) = \Sigma_{i=1}^nf(x_i;\theta)
$$

#### 逻辑回归原理

对于一个二分类问题，我们假设样本服从一个伯努利分布，即$X\sim B(1,p)$，现在有样本$(\mathbf{x_i},y_i)$，其中，$y_i$ 是类别，只有两种取值，即0和1，我们假设概率$p$ 与样本点的关系如下：
$$
p = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}
$$
之所以这么选取是因为sigmoid函数的取值为$(0,1)$,则似然函数为
$$
L(\mathbf{w}) = \Pi_{i=1}^n(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}})^{y_i}(1-\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}})^{1-y_i}\\
=\Pi_{i=1}^n(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}})^{y_i}(\frac{e^{-\mathbf{w}^T\mathbf{x}}}{1+e^{-\mathbf{w}^T\mathbf{x}}})^{1-y_i}
$$
为了方便，对上述式子取对数
$$
lnL(\mathbf{w}) = \Sigma_{i=1}^n-y_i*ln(1+e^{-\mathbf{w}^T\mathbf{x}})+(1-y_i)*(-\mathbf{w}^T\mathbf{x}-ln(1+e^{-\mathbf{w}^T\mathbf{x}}))
$$
因为梯度下降需要损失函数减小，所以可以对上述似然函数取负作为损失函数
$$
Loss = -lnL(\mathbf{w}) =\Sigma_{i=1}^ny_i*ln(1+e^{-\mathbf{w}^T\mathbf{x}})+(1-y_i)*(\mathbf{w}^T\mathbf{x}+ln(1+e^{-\mathbf{w}^T\mathbf{x}}))=(1-y_i)*\mathbf{w}^T\mathbf{x}+ln(1+e^{-\mathbf{w}^T\mathbf{x}}))
$$
求梯度
$$
\frac{dLoss}{d\mathbf{w}} = \Sigma_{i=1}^n(1-y_i)*\mathbf{x}+\frac{-e^{-\mathbf{w}^T\mathbf{x}}}{1+e^{-\mathbf{w}^T\mathbf{x}}}*\mathbf{x} = \Sigma_{i=1}^n(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}-y_i)*\mathbf{x}
$$
运用梯度下降
$$
w_{t+1} = w_{t} - \eta*\Sigma_{i=1}^n(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}-y_i)*\mathbf{x}
$$
运用随机梯度下降
$$
w_{t+1} = w_{t} - \eta*(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}-y_i)*\mathbf{x}
$$

* 决策边界

一般选取0.5为决策边界，即$y_i$ 为1的概率$p$ 超过了0.5即视作属于该类

* 损失函数的选取

在选择损失函数时，应尽量保证函数的凸性，即对一个凸函数求最小值，利用最大似然推导出来的损失函数恰好是凸函数，所以可用，在这里，如果用的是F范数作为损失函数，结果如何？

可以检验一下是不是凸函数，吴恩达说不是

### 8. 分类问题中的那些率

#### 二分类问题的可能结果

|       | 预测1                        | 预测0                        | 合计                      |
| ----- | ---------------------------- | ---------------------------- | ------------------------- |
| 真实1 | TP                           | FN                          | TP+FN （Actual Positive） |
| 真实0 | FP                           | TN                          | FP+TN （Actual Negative） |
| 合计  | TF+FP （Predicted Positive） | FN+TN （Predicted Negative） ||

上面说到，在逻辑回归训练完毕之后，测试时，输入特征$\mathbf{x}$ ，会输出一个概率$p$， 那么概率到达多少可以作为判断分类为$1$的依据呢，这是一个值得考虑的问题，选取的概率值可以称之为决策面，决策面的选取导致了第一类错误和第二类错误的发生。

* 第一类错误

漏检：原假设为真，缺排除了原假设，即真实1，预测0

* 第二类错误

虚警：原假设为假，缺认为为真，即真实0，预测1

在战争年代，第二类错误更可怕，容易引起世界大战。

#### 各种率和ROC曲线

* 准确率（ACC）

$$
\text{准确率} = \frac{\text{分类正确}}{\text{总样本数}}
$$

常见指标，最容易理解，但是在正负样本不均衡的时候说服力低，比如人群中奢侈品消费人数很少，数据收集过来是小样本，这样训练出来的广告推荐系统可能准确率很高，但对奢侈品销量没有影响，很可能是奢侈品人群数据在样本占比很低，导致输出推荐全为普通人群商品也可以获得很好的准确率。

* 精准率-查准率-精确率(Precision)

$$
P = \frac{TP}{TP+FP}
$$

预测为正的样本中，有多少比例是真的正

* 召回率，查全率(Recall)

$$
TPR = \frac{TP}{TP+FN}
$$

正样本数被查出来的比例，所以又叫查全率



* 虚警率(FP)

$$
FPR = \frac{FP}{TN+FP}
$$

输出为1，但实际为0的样本占所有0样本的比例

召回率和虚警率是对所有预测为正的样本的划分和分析。

* ROC曲线

ROC曲线的纵坐标是召回率，横坐标是虚警率，因此ROC曲线上有四个特殊的点：

（0,1）：所有样本分类正确

（1,0）：所有样本分类错误

（0,0）：所以样本都分类为负

（1:1）：所以样本都分类为正

绘制方法，不断地去移动决策边界，得到一组(recall,fpr)，在图上绘制一个点，从而得到一个阶梯ROC曲线

#### ROC曲线的评估方法

* AUC

AUC是ROC曲线下的面积，物理意义可以按如下进行理解：**从所有1样本中随机选取一个样本， 从所有0样本中随机选取一个样本，然后根据你的分类器对两个随机样本进行预测，把1样本预测为1的概率为p1，把0样本预测为1的概率为p0，p1>p0的概率就等于AUC**

1. AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。

2. 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。

3. AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。

4. AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

不同分类器的ROC曲线无交叠时，AUC面积越大越好

![img](https://pic4.zhimg.com/80/v2-4b20e30bd7d8a1d00477b3b090d36e43_720w.jpg)

* 曲线比较方法

![img](https://pic3.zhimg.com/80/v2-3711e0fb080735058b1c3c88778d0eb2_720w.jpg)

当AUC面积几乎一样，而ROC又有交叉时，可以根据需要进行选择，需要较高的Recall(sensitivity)，则选择A，需要较低的虚警，则选择B

* 最优临界点

<img src="https://pic3.zhimg.com/80/v2-6efbdbd084de9fda0a477ef7e1133ff6_720w.jpg" alt="img" style="zoom:80%;" />

ROC曲线上的最优临界点，让Recall尽量高的情况下，不显著增加TPR，这个点是距离(0,1)最近的点

### 9.多分类问题的解决模式

#### 逻辑斯谛分布

$$
\begin{aligned}
F(x) = P(X\leq x) = \frac{1}{1+e^{-(x-\mu)/\gamma}}\\
f(x) = F^{'}(x) = \frac{e^{-(x-\mu)}}{\gamma(1+e^{-(x-\mu)/\gamma})}
\end{aligned}
$$

假设样本服从二项逻辑斯谛分布，就是LR，用于二分类问题，假设样本服从多项逻辑斯谛分布，就是mlr，对应的激活函数就是softmax，损失函数就是交叉熵。

### 10. SVM的推导

#### 线性可分支持向量机

对于一批训练数据$\Tau = \{(\mathbf{x}_i,y_i)\}_{i=1}^N$​​ ​，其中$y_i = 1(positive);y_i=-1(negative)$， 现在想要找到一个最优的决策面，$\mathbf{w}^T\mathbf{x}+b=0$,使得
$$
\begin{aligned}
\mathbf{w}^T\mathbf{x_i}+b>=0 ,\ for \ y_i=1\\
\mathbf{w}^T\mathbf{x_i}+b<0 ,\ for \ y_i=-1\\
\end{aligned}
$$
 如何利用已有的数据，得到最优的$\mathbf{w}$和$b$ 就是SVM的核心思想，定义如下的优化问题：
$$
\begin{aligned}
\max &\frac{2}{\|\mathbf{w}\|_2^2}\\
s.t. &\mathbf{w}^T\mathbf{x}_j+b\geq1,\ y_j = 1\\
&\mathbf{w}^T\mathbf{x}_k+b\leq-1,\ y_k=-1
\end{aligned}
$$
这样是把点到直线的距离转化成了直线与直线的距离，最优的决策面仍然为$\mathbf{w}^T\mathbf{x}+b=0$​，构造两条平行线作为“支撑向量”的边界，两条平行线之间的距离叫做间隔，上述问题等价于
$$
\begin{aligned}
\min\  &\frac{1}{2}\|\mathbf{w}\|_2^2\\
s.t.\  &y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq1 \ ,i=1,\cdots,N
\end{aligned}
$$
该问题是一个凸二次规划问题，可以直接求解，且具有唯一解，但是为了进一步分析，我们求它的对偶问题，该问题的拉格朗日函数为
$$
L(\boldsymbol{\alpha},b,\mathbf{w}) =\frac{1}{2}\|\mathbf{w}\|_2^2 + \Sigma_{i=1}^{N}\alpha_i(1-y_i(\mathbf{w}^T\mathbf{x}_i+b))
$$
为了求解拉格朗日函数的下确界，对$\mathbf{w},b$​​​ 求偏导等于0，得到
$$
\mathbf{w} = \Sigma_{i=1}^N\alpha_iy_i\mathbf{x}_i,\Sigma_{i=1}^N\alpha_iy_i=0
$$
代入原式，得到对偶问题
$$
\begin{aligned}
\max \inf \ L(\boldsymbol{\alpha},b,\mathbf{w}) & = \max -\frac{1}{2}\Sigma_{i=1}^{N}\Sigma_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf{x_i}^T\mathbf{x}_j + \Sigma_{i=1}^N\alpha_i\\
&= \min\  \frac{1}{2}\Sigma_{i=1}^{N}\Sigma_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf{x_i}^T\mathbf{x}_j -\Sigma_{i=1}^N\alpha_i\\\
s.t. \ &\Sigma_{i=1}^N\alpha_iy_i = 0,\\ 
&\alpha_i\geq0,i=1,\cdots,N
\end{aligned}
$$

##### 支持向量

对于线性可分的情况，支持向量有两种定义：

* 训练样本中与分离超平面距离最近的样本点的实例
* 训练样本中对应$\alpha^*_i>0$的点

由凸优化KKT中的互补松弛条件，二者等价。

#### 线性支持向量机

对于一个问题，大部分样本点满足线性可分，少部分点不满足线性可分，我们采用软间隔方法，也称作线性支持向量机，定义优化问题如下：
$$
\begin{aligned}
\min\  &\frac{1}{2}\|\mathbf{w}\|_2^2 + C\Sigma_{i=1}^N\xi_i\\
s.t.\  &y_i(\mathbf{w}^T\mathbf{x}_i+b)\geq1-\xi_i \ ,i=1,\cdots,N\\
&\xi_i\geq0,i=1,\cdots,N
\end{aligned}
$$
其中，$C$为超参数，代表对误分类的惩罚，可以证明，上述问题为凸二次规划问题，有最优解，其中$\mathbf{w}$ 是唯一的，$b$ 是一个区间。

上述问题的对偶问题为
$$
\begin{aligned}
\min\ & \frac{1}{2}\Sigma_{i=1}^{N}\Sigma_{j=1}^N\alpha_i\alpha_jy_iy_j\mathbf{x_i}^T\mathbf{x}_j -\Sigma_{i=1}^N\alpha_i\\\
s.t. \ &\Sigma_{i=1}^N\alpha_iy_i = 0,\\ 
&0 \leq \alpha_i \leq C,i=1,\cdots,N
\end{aligned}
$$
线性支持向量机的支持向量指的是所有在间隔边界上以及其上方的点，对应对偶问题中$\alpha^*_i>0$ 的点：

* $\alpha_i<C,\xi_i=0$,落在了间隔边界上
* $\alpha_i=C,0<\xi<1$,落在间隔边界和分类边界之间
* $\alpha_i=C,\xi=1$​, 落在了分类边界上
* $\alpha_i=C,\xi>1$，落在了分类错误的那边

#### 非线性支持向量机

对于非线性问题，先利用一个非线性变换将输入空间对应到一个特征空间，使得输入空间的超曲面模型在特征空间中变成超平面模型，然后使用线性支持向量机求解。

##### 核函数

设一个输入空间到特征空间的映射为$\Phi(x)：\mathcal{X}->\mathcal{H}$​​​,对任意的$x,z \in \mathcal{X}$，函数$K(x,z)=\Phi(x)\Phi(z)$，则$K(x,z)$称为核函数，$\Phi(x)$​称为映射函数。由于映射函数比较难以获得求解，在这里主要使用核函数，而核函数的选择较为简单，只要一个定义在$\mathcal{X}\times\mathcal{X}$上的对称函数满足核矩阵半正定即可，核矩阵定义为任意数据输入到对称函数的互相关矩阵。常用的核函数有：

* 线性核

$$
K( \mathbf{x},\mathbf{z}) = \mathbf{x}^T\mathbf{z}
$$

* 多项式核

$$
K( \mathbf{x},\mathbf{z})  = ( \mathbf{x}^T\mathbf{z}+1)^p
$$



* 高斯核（最常用）

$$
K( \mathbf{x},\mathbf{z})  = exp(-\frac{\| \mathbf{x}- \mathbf{z}\|^2}{2\sigma^2})
$$



* sigmoid核

$$
K( \mathbf{x},\mathbf{z})  = tanh(\beta  \mathbf{x}^T\mathbf{z}+\theta)
$$

* 拉普拉斯核

$$
K( \mathbf{x},\mathbf{z})  =exp(-\frac{\| \mathbf{x}- \mathbf{z}\|}{\sigma})
$$

* 字符串核

##### 问题形式

$$
\begin{aligned}
\min\ & \frac{1}{2}\Sigma_{i=1}^{N}\Sigma_{j=1}^N\alpha_i\alpha_jy_iy_jK(\mathbf{x_i},\mathbf{x}_j) -\Sigma_{i=1}^N\alpha_i\\\
s.t. \ &\Sigma_{i=1}^N\alpha_iy_i = 0,\\ 
&0 \leq \alpha_i \leq C,i=1,\cdots,N
\end{aligned}
$$

#### 支持向量机的求解——SMO算法

优化时固定其他变量不变，每次只优化两个变量，使问题变成两变量的二次优化问题，直到求得的解在误差范围内满足KKT条件。

#### 分类问题的方法选取

* feature数目和样本数目差不多，线性可分，逻辑回归或者线性SVM
* 样本数量很多，高斯核运算慢，可以手动增加feature，使其变得线性可分
* 样本数量不算多，但feature少，用SVM+高斯核函数

### 11.1x1卷积的作用

* 基本原理

1x1的卷积核使得在一张feature map上，所有的特征（元素）等比例的扩大或者缩小：

1. 当channel = 1时，基本没有作用，因为只是所有特征乘了一个系数
2. 当channel>1时，输出的值相当于在channel维度上对所有的特征做了叠加

因此，常用来升维和降维，调整特征图的深度，有一个假设是特征图是过冗余的，所以可以在不丢失信息的前提下通过1x1卷积降维；具体与全连接的关系，在当前的深度学习框架下，对于channel_last的特征图，1x1卷积和全连接没有区别

### 12. 常用深度学习优化器原理

### 13. 信息熵、交叉熵、相对熵

* 信息熵

信息熵是随机变量的不确定度的度量
$$
H(P) = -\Sigma_{i=1}^np_ilog(p_i)
$$

* 交叉熵

用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小
$$
H(P,Q) = -\Sigma_{i=1}^np_ilog(q_i)
$$

* 相对熵

用来衡量两个分布的差异，其中P是真实分布，Q是估计的分布
$$
D(P||Q) = H(P,Q)-H(P) =\Sigma_{i=1}^np_ilog(\frac{p_i}{q_i})
$$
一般来说，相对熵是非对称的，即$D(P||Q)!=D(Q||P)$



### 14. MSE，方差，偏差

<img src="C:\Users\Ensiwalk\Documents\GitHub\Interview-Kit\image-20210721233335981.png" alt="image-20210721233335981" style="zoom:50%;" />

对一个估计量而言

* 方差：描述了预测值的离散程度，方差越大，离散程度越高

$$
var(\hat{\theta}) = \mathbb{E}((\hat{\theta}-\mathbb{E}(\theta)^2)
$$




* 偏差：描述了预测值的准确程度，偏差越大，预测越不准确

$$
bias(\hat{\theta}) = \mathbb{E}(\hat{\theta}) - \theta
$$



当偏差为0时，是一个无偏估计，对于两个无偏估计而言，方差越小越好，这样估计的结果较为稳定。

* MSE：均方误差，估计值与参数的差的平方取均值

$$
MSE = \mathbb{E}((\hat{\theta}-\theta)^2)  = var(\hat{\theta})+bias^2(\theta)
$$

### 15. 聚类方法

#### 原型聚类

##### K-means

##### LVQ

##### 高斯混合

#### 密度聚类

#### 层次聚类

### 16. 降维方法



### 17. 决策树

#### 基本算法

![img](https://pic4.zhimg.com/80/v2-c64ce8e323e1996b4f9c943a91d47934_hd.jpg)

决策树是递归算法，递归地建立一棵树，有三个递归返回方式：

（1）所有的样本都是同一类C，无需继续划分，标记为C的叶子节点

（2）当前的属性集为空，或者所有样本在所有的属性上都是相同的，此时把当前节点标记为叶子节点，投票选出包含样本最多的类别C作为叶子节点的类别

（3）当前节点包含的样本集合为空，标记为叶子节点，将父亲节点中包含最多的类别C作为当前节点的类别

#### ID3算法——信息增益法：表示特征对于不确定性减少的程度

在第八步，选最优属性时，有不同的方法，对应了不同的算法。
$$
Ent(D) = -\Sigma_{k=1}^{K}p_klog_2(p_k)
$$
对于一个离散属性，$a = \{a^1,\cdots,a^V\}$共$V$​个取值，用这些，取值对数据集做一个划分，对任意一个子集$D^v$,计算一下信息熵$Ent(D^v)$，则信息增益可以进行如下定义：
$$
Gain(D,a) = Ent(D) - \Sigma_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)
$$
计算出当前数据集在不同的属性上的信息增益后
$$
a^* = argmax Gain(D,a)
$$

#### C4.5算法——信息增益率

C4.5算法选择了信息增益率作为了划分依据，信息增益率定义如下
$$
\begin{aligned}
Gain\_ratio = \frac{Gain(D,a)}{IV(a)}\\
IV(a) = -\Sigma_{v=1}^V\frac{|D^v|}{|D|}log_2{\frac{|D^v|}{|D|}}
\end{aligned}
$$
通常，a的取值越多，$IV(a)$的值偏向于越大，因此增益率对于可取属性值较少的属性有偏好，实际操作时，可以先选出$IV$​较大的属性，再从中选择增益率最大的属性。

#### CART——基尼指数

$$
Gini(D) = \Sigma_{k=1}^K\Sigma_{k^\prime \neq k}p_k p_k^{\prime} = 1-\Sigma p_k^2
$$

直观上看，从一组样本中任取两个样本，二者不一样的概率越小，说明数据纯度越高，对属性D，基尼指数可以定义为
$$
Gini\_index(D,a) = \Sigma_{v=1}^V \frac{|D^v|}{|D|}Gini(D^v)
$$

$$
a^* = argmin Gini\_index(D,a)
$$



#### 剪枝处理

剪枝是预防决策树算法过拟合的方法，当分支过多时，很可能出现训练集完全拟合，但测试时效果很差，此时便需要剪掉一些支路简化决策树，提高泛化性。剪枝利用验证集进行。

* 预剪枝

预剪枝是指每划分一个节点时，都计算一下划分前后的收益，即不划分该节点，验证集得到的准确率是a，划分该节点验证集得到的准确率是b，如果a>b，则不进行划分了，本质上是一种贪心算法。

预剪枝使得决策树没有展开，降低了过拟合风险，但由于是贪心，所以很可能不是最优的，类似viterbi译码和贪心译码。

* 后剪枝

后剪枝是对一棵决策树自底向上，对所有的非叶子节点进行考察，看它变成叶子节点能否提升泛化性，可以的话就变。

#### 连续值的处理

连续值进行二分法处理，例如属性a在数据D中出现了n个值，则有一个候选分割点集合
$$
T_a = \{\frac{a^i,a^{i+1}}{2},1\leq i \leq n-1 \}
$$

$$
Gain(D,a) = max\  Gain(D,a,t) = max\ Ent(D)-\Sigma \frac{D_t}{D}Ent(D_t)
$$

#### 缺失值的处理

其实思路很简单，对于某个特征，先找出没有缺失的数据集$\tilde{D}$,计算一个系数
$$
\rho = \frac{|\tilde{D}|}{|D|}
$$
然后在$\tilde{D}$上进行计算信息增益，计算完毕后，再乘上系数$\rho$，以此类推

### 18. 集成学习和Gbdt

集成学习通过构建并结合多个学习器来完成学习任务。可以证明，对于性能相同的相互独立的多个学习器进行简单投票集成，可以使错误率指数下降趋于0，但是独立这个假设很难达到。

个体学习器之间是串行关系，有强依赖，用Boosting算法集成；个体学习器之间是并行关系，用Bagging/随机森林算法集成。

### 20. 机器学习和深度学习的对比

* 数据依赖性

深度学习需要的数据更多，在大数据的情况下，达到的性能更好

* 硬件依赖

深度学习需要大量矩阵运算，可以用GPU加速

* 特征处理

机器学习大多是专家选取，深度学习大多是数据中提取的

* 解决方式

深度学习是端到端的，机器学习一般拆解为子问题

* 执行时间

深度学习训练时间很长，执行复杂度较低

* 可解释性差

深度学习很难解释，仅有的理论也只是收敛性上的，很少有能从参数、结构设计上有可以被证明的结论；机器学习的决策树，SVM等算法解释性很强



## 凸优化篇

### 0.凸问题



### 1. 对偶及其理解

#### 原对偶问题

对一个优化问题
$$
\begin{aligned}
\min&\  f_0(x)\\
s.t.&\ f_i(x)\leq0\ \ for \ i=1,\cdots,m\\
&\ h_i(x)=0\ for\ i=1,\cdots,p
\end{aligned}
$$
其拉格朗日函数可以写作，
$$
\begin{aligned}
L = f_0(x)+\Sigma\lambda_if_i(x)+\Sigma\mu_ih_i(x)\\
g(\boldsymbol{\lambda},\boldsymbol{\mu}) = \inf L
\end{aligned}
$$
则对偶问题为
$$
\begin{aligned}
\max g(\boldsymbol{\lambda},\boldsymbol{\mu})\\
s.t. \boldsymbol{\lambda}\geq0\\
dmog=\{(\lambda,\mu)|g(\boldsymbol{\lambda},\boldsymbol{\mu})>-\infty\}
\end{aligned}
$$

#### 对偶性的强度

* 弱对偶

`弱对偶` $d^*$为对偶问题的最优解，$p^*$ 为原问题的最优解，若$d^*\leq p^*$，则为弱对偶，$p^*-d^*$​为对偶间距。

* 强对偶

`强对偶` $d^*=p^*$

强对偶的条件很苛刻，目前已知有三种情况是成立的：

1）凸问题+slater条件成立

2）有可行域的线性规划问题

3）二次约束的二次目标问题+slater成立

* 对偶的几何性解释

### 2. Slater 和 KKT



### 3. 凸问题的求解

#### 无约束优化

##### 下降法

##### 最速下降法

##### 梯度下降法

##### 牛顿法

#### 不等式约束优化

##### 对数障碍法

##### 原对偶内点法

## 通信原理及通信信号处理篇

