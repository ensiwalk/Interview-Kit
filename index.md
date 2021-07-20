<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
</head>


## 机器学习篇

### 1. Batch Normalization

#### 基本公式
$$\mu = \frac{1}{m}\Sigma_{i=1}^mx_i\\
\sigma^2 = \frac{\Sigma_{i=1}^m(x_i-\mu)^2}{m}\\
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2 + \epsilon}}\\
y_i = \gamma\hat{x}_i + \beta
$$

\\(\epsilon\\) 是为了防止方差为0，\\(\gamma\\)  和  \\(\beta\\)是可学习参数，为了使BN后的数据仍保留一定的原有特征，因为二者选择的比较好，可以使处理后的数据回归原始数据。

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
\frac{d f_2}{d w} = \frac{d f_2}{d f_1}x
$$

可以看到，求梯度的时候有一项与输入\\(x\\) 有关，而BN消除了\\(x\\) 缩放带来的影响

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
* 数据增强，增加少数样本，mix-up，旋转，反向，加噪，相位等等

### 6. 分类问题中的那些率

|       | 预测1                        | 预测0                        | 合计                      |
| ----- | ---------------------------- | ---------------------------- | ------------------------- |
| 真实1 | TP                           | TN                           | TP+FN （Actual Positive） |
| 真实0 | FP                           | FN                           | FP+TN （Actual Negative） |
| 合计  | TF+FP （Predicted Positive） | FN+TN （Predicted Negative） |                           | 

### 7. 逻辑回归推导及其正向反向传播
#### 最大似然估计

最大似然估计的基本思想：拥有一组数据的样本观测值，并且已知其含参数概率分布的形式，选取合适的参数估计值，使得样本取到样本值的概率最大。

* 样本

设\\((X_1,X_2,\cdots,X_n)\\)是来自总体的一个容量为n的样本，\\((x_1,x_2,\cdots,x_n)\\) 是相应的样本值。

* 离散型总体的似然函数

设分布律为\\(P(X=x) = p(x;\theta),\theta \in \Theta\\)的形式已知，\\(\theta\\)为待估参数，则样本的观察值为\\((x_1,x_2,\cdots,x_n)\\) 的概率为

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

设$X \sim f(x;\theta)$，$\theta$ 为待估参数，则上述样本的联合概率密度为

$$
L(\theta) = L(x_1,\cdots,x_n;\theta) = \Pi_{i=1}^nf(x_i;\theta)
$$

因为样本在样本点附近取值为大概率事件，所以要最大化上述似然函数

对数似然函数为

$$
lnL(\theta) = \Sigma_{i=1}^nf(x_i;\theta)
$$

#### 逻辑回归原理

对于一个二分类问题，我们假设样本服从一个伯努利分布，即\\(X\sim B(1,p)\\)，现在有样本\\((\mathbf{x_i},y_i)\\)，其中，\\(y_i\\) 是类别，只有两种取值，即0和1，我们假设概率\\(p\\) 与样本点的关系如下：

$$
p = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}
$$

之所以这么选取是因为sigmoid函数的取值为\\((0,1)\\),则似然函数为

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
Loss = -lnL(\mathbf{w}) =\Sigma_{i=1}^ny_i*ln(1+e^{-\mathbf{w}^T\mathbf{x}})+(1-y_i)*(\mathbf{w}^T\mathbf{x}+ln(1+e^{-\mathbf{w}^T\mathbf{x}}))\\
=(1-y_i)*\mathbf{w}^T\mathbf{x}+ln(1+e^{-\mathbf{w}^T\mathbf{x}}))
$$

求梯度

$$
\frac{dLoss}{d\mathbf{w}} = \Sigma_{i=1}^n(1-y_i)*\mathbf{x}+\frac{-e^{-\mathbf{w}^T\mathbf{x}}}{1+e^{-\mathbf{w}^T\mathbf{x}}}*\mathbf{x}\\
= \Sigma_{i=1}^n(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}-y_i)*\mathbf{x}
$$

运用梯度下降

$$
\mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta*\Sigma_{i=1}^n(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}-y_i)*\mathbf{x}
$$

运用随机梯度下降

$$
\mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta*(\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}-y_i)*\mathbf{x}
$$

### 8. 1x1卷积的作用


