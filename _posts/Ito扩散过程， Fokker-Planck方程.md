# 介绍
## Ito扩散过程与反向过程

- Ito扩散过程可以由以下SDE来描述：$d \boldsymbol{x} = \boldsymbol{f} (\boldsymbol{x}, t)dt + \boldsymbol{G}(\boldsymbol{X}, t)d\boldsymbol{w}$，其中 $\mathbf{f}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^d$ ， $\mathbf{G}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^{d \times d}$。
- 反向过程由以下SDE描述：
$\mathrm{d} \mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} \mathrm{d} t+\mathbf{G}(\mathbf{x}, t) \mathrm{d} \overline{\mathbf{w}}$

   - $\mathbf{F}(\mathbf{x}):=\left(\mathbf{f}^1(\mathbf{x}), \mathbf{f}^2(\mathbf{x}), \cdots, \mathbf{f}^d(\mathbf{x})\right)^{\top}$是一个矩阵方程$\nabla \cdot \mathbf{F}(\mathbf{x}):=\left(\nabla \cdot \mathbf{f}^1(\mathbf{x}), \nabla \cdot \mathbf{f}^2(\mathbf{x}), \cdots, \nabla \cdot \mathbf{f}^d(\mathbf{x})\right)^{\top}$ 
# 离散时间马尔可夫过程
## 什么是马尔可夫过程

- 首先简单了解一下离散时间马尔可夫过程。简单理解，马尔可夫过程可以看作是系统随时间随机演变的状态。当前状态确定了之后，未来状态和过去状态就相互独立了。
- 马尔可夫过程定义为一系列随机变量$X_0, X_1, X_2, \cdots$， 具有这样的性质：给定一个特定的随机变量，其左侧的变量与其右侧的变量无关。即：$X_0,X_1,\ldots,X_{i-1}\perp X_{i+1},X_{i+2},\ldots|X_i$
## 马尔可夫过程的性质

- 马尔可夫过程的条件概率可以得到简化: $P(X_{i+1} \vert X_i=x_i, \cdots ,X_1=x_1,X_0=x_0) = P(X_{i+1} \vert X_i = x_i)$
- 马尔可夫过程的整个序列的概率由初始状态的分布和以及转移概率来确定。$P(X_0=x_0,X_1=x_1,X_2=x_2,\ldots)=P(X_0=x_0)\prod_{t\in\{1,2,\ldots\}}P(X_t=x_t|X_{t-1}=x_{t-1})$
- 注意转移概率在不同时间可能是不同的。根据马尔可夫的特性，我们可以很容易得到以下递归性质。$P(X_t=x_t|X_s=x_s)=\sum_kP(X_t=x_t|X_m=k)P(X_m=k|X_s=x_s)$
- 如果$X_0, X_1,  \cdots X_T$是马尔可夫过程，那么反过来$X_T, X_{T-1},  \cdots X_0$也是马尔可夫过程。简单理解，给定当前状态，未来与过去独立的，那么反过来说，给定当前状态，过去也与未来是独立的。
- 全概率公式（不依赖于马尔可夫特性）：$P(X_t=x_t)=\sum_kP(X_t=x_t|X_m=k)P(X_m=k)$
# 柯尔莫哥洛夫方程
## 连续时间的马尔可夫过程

- 如果系统状态是连续的，那么我们可以对马尔可夫过程进行推广。递归性质里面的求和要变成积分：$p(x;t|y;s)=\int_{-\infty}^\infty p(x;t|k;m)p(k;m|y;s)\mathrm{d}k$。这里$p(x;t|y;s)$表示概率密度$p(X_t=x|X_s=y)$，表示时刻$s$时状态$y$转移到时刻$t$时状态$x$的概率密度，也就是说，这个条件概率密度是四个变量$s, y, t, x$的函数。
- 这里同样有一个不依赖马尔可夫特性的全概率公式：$p(x;t)=\int_{-\infty}^\infty p(x;t|k;m)p(k;m)\mathrm{d}k$

## 前向柯尔莫哥洛夫方程
柯尔莫哥洛夫方程或者福克普朗克方程，描述的是概率密度的演化

- 首先考虑在很短的时间内条件概率密度的演化：
- $p(x;t + dt|y,s) = \int_{-\infty}^{\infty} p(x; t + dt | m;t) p(m;t |y;s)dm$
- 这里计算的核心在两项：$p(x; t + dt | m;t)$ 和$p(m;t |y;s)$，其中$p(x; t + dt | m;t)$描述的变化可能非常小，也就是说$x$和$m$之间非常接近。这里考虑一个换元的技巧，将$x$写成$x = m + \Delta$，那么$m = x - \Delta$; $dm = -d \Delta$,  $m = \pm \infty \Rightarrow \Delta =  \mp \infty$
- $$\begin{align}

p(x;t + dt|y,s) &= \int_{-\infty}^{\infty} p(x; t + dt | m;t) p(m;t |y;s)dm \\
&= \int_{-\infty}^{\infty} p(m + \Delta; t + dt | m;t) p(m;t |y;s)d\Delta \\

\end{align}$$
- 定义 $\phi_t(\Delta; z) = p(z+\Delta; t+dt|z;t)$，上式可写为$p(x;t + dt|y,s) = \int_{-\infty}^{+\infty} \phi_t(\Delta ; m) p(m ; t \mid y ; s) \mathrm{d} \Delta$
- 现在我们可以对$\phi_t (\Delta; m)$关于$x$泰勒展开
- $\begin{aligned} p(x ; t+\mathrm{d} t \mid y ; s) & =\int_{-\infty}^{+\infty} \phi_t(\Delta ; x) p(x ; t \mid y ; s) \mathrm{d} \Delta \\ & -\int_{-\infty}^{+\infty} \Delta \frac{\partial}{\partial x} \phi_t(\Delta ; x) p(x ; t \mid y ; s) \mathrm{d} \Delta \\ & +\int_{-\infty}^{+\infty} \frac{\Delta^2}{2} \frac{\partial^2}{\partial x^2} \phi_t(\Delta ; x) p(x ; t \mid y ; s) \mathrm{d} \Delta\end{aligned}$
- $\phi_t(\Delta; x)$关于$\Delta$的积分为1。我们交换一下积分和偏微分的顺序：
- $$\begin{aligned}
\begin{aligned}p(x;t+\text{d}t|y;s)-p(x;t|y;s)\end{aligned}& =-\frac\partial{\partial x}\big(\mathbb{E}_{\Delta\sim\phi_t(;x)}[\Delta]p(x;t|y;s)\big) \\
&+\frac12\frac{\partial^2}{\partial x^2}\big(\mathbb{E}_{\Delta\sim\phi_t(;x)}\big[\Delta^2\big]p(x;t|y;s)\big) \\
& + \cdots
\end{aligned}$$
- 现在我们将级数截断为第二项，定义$\mathbb{E}_{\Delta\sim\phi_t(;x)}[\Delta]:=f(x,t)\mathrm{d}t$，$\mathbb{E}_{\Delta\sim\phi_t(;x)}\left[\Delta^2\right]:=g^2(x,t)\mathrm{d}t$，两边除以$dt$我们得到了微分方程$\frac{\partial}{\partial t} p(x ; t \mid y ; s)=-\frac{\partial}{\partial x}(f(x, t) p(x ; t \mid y ; s))+\frac{1}{2} \frac{\partial^2}{\partial x^2}\left(g^2(x ; t) p(x ; t \mid y ; s)\right)$
- 上面的$\phi_t(;X)$正是$dX$的分布。$dX \sim \phi_t(;X)$, $E[dX] = f(X, t)dt$, $Var(dX) = g^2(X, t)dt - O(dt^2) \approx g^2(X, t) dt$。函数$f$称为扩散过程的漂移系数，$g$称为扩散系数。据此，我们可以将$dX$写成以下形式：$\mathrm{d}X=f(X,t)\mathrm{d}t+g(X,t)\mathrm{d}w$
- 这里$dw$是一个方差为dt并且均值为0的分布，并且与当前以及过去的状态相互独立。这个非常就是大名鼎鼎的Ito SDE。实际上这里的分布就只能是高斯分布了，这里不做详细证明。可以想象一下，任取一小段时间$\Delta t$, 总可以将$\Delta t$分成很多的更小的片段，比如说分成$M$段，这些每一个小的片段都是均值为0，方差为$\Delta t / m$，根据中心极限定理，这些小段加在一起就服从均值为0，方差为$\Delta t$ 的高斯分布了。

### 反向柯尔莫哥洛夫方程
反向柯尔莫哥洛夫方程跟前向的类似，我们可以推导出类似的泰勒展开
$\begin{aligned} p(x ; t \mid y ; s-\mathrm{d} s) & =\int_{-\infty}^{+\infty} \phi_{s-\mathrm{d} s}(\Delta ; y) p(x ; t \mid y+\Delta ; s) \mathrm{d} \Delta \\ & =\int_{-\infty}^{+\infty} \phi_{s-\mathrm{d} s}(\Delta ; y) p(x ; t \mid y ; s) \mathrm{d} \Delta \\ & +\int_{-\infty}^{+\infty} \phi_{s-\mathrm{d} s}(\Delta ; y) \Delta \frac{\partial}{\partial y} p(x ; t \mid y ; s) \mathrm{d} \Delta \\ & +\int_{-\infty}^{+\infty} \phi_{s-\mathrm{d} s}(\Delta ; y) \frac{\Delta^2}{2} \frac{\partial^2}{\partial y^2} p(x ; t \mid y ; s) \mathrm{d} \Delta \\ & \vdots\end{aligned}$

反向柯尔莫哥洛夫方程：$-\frac{\partial}{\partial s} p(x ; t \mid y ; s)=f(y, s) \frac{\partial}{\partial y} p(x ; t \mid y ; s)+\frac{g^2(y ; s)}{2} \frac{\partial^2}{\partial y^2} p(x ; t \mid y ; s)$

# Reference
[https://www.vanillabug.com/posts/sde/](https://www.vanillabug.com/posts/sde/)
