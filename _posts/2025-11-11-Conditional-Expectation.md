---
layout: post
title: "Conditional Expectation - the core of diffusion models"
date: 2025-11-11
tags: [reading notes]
---
The conditional expectation is the core to connect almost all things in diffusion models, e.g., the sampling ODE, SDE, training objective, score function and so on. This note try to understand diffusion models by conditional expecation.

# Preliminary

**Conditional expectation**. $\mathbb{E}[f(x_0) \mid x_t]:= \int f(x_0) p( x_0\mid x_t) dx_0$.

**Diffusion model**. Given training dataset $\mathcal{D}=\{ x_0^{i} \}_{i=1}^N$ from target distribution $\pi_0(x_0)$, $x_0^{i} \in \mathbb{R}^d$, the goal of generative modelling is to draw new samples from $\pi_0$. Generally, it's hard to directly sample from the target distribution $\pi_0$, since it is usually complex. But we can build a diffusion process to transfer a simple prior density $\pi_1$ to target density $\pi_0$ gradually, where boundary conditions are required: $p_0=\pi_0, \quad p_1 = \pi_1$. Usually we select $\pi_1$ as a Gaussian distribution: $\pi_1\sim \mathcal{N} (0, \mathbb{I})$.

# **Design of the diffusion process**

The problem is how to design $p_t$? To answer this question, we first introduce conditional flow $p_{t \mid 0} (x_t \mid x_0)$, which is the conditional distribution of $x_t$ given $x_0$. Then we can $p_t$ as a mixture of densities: $p_t(x_t) = \int p_t(x_t \mid x_0) \pi_0(x_0) dx_0$. If we let the transition kernel $p_t(x_t \mid x_0)$ to be a Gaussian: $p_{t \mid 0} (x_t \mid x_0) = \mathcal{N} (x_t; \alpha_t x_0, \sigma_t^2 \mathbb{I})$, then the distribution $p_t$ is smoother as $t$ grows, and $p_t$ is given by:

$$\begin{align}p_t(x_t) &= \int p_t (x_t \mid x_0) \pi_0(x_0) dx_0\\ & = \int \mathcal{N} (x_t; \alpha_tx_0, \sigma_t^2 \mathbb{I}) \pi_0(x_0) dx_0  \end{align} \tag{1}$$

As we set $\alpha_0=1$, $\sigma_0\rightarrow0$, $\alpha_1 = 1$, $\sigma_1 = 1$. $p_t$ is a diffusion process that transfer from data distribution $p_0=\pi_0$ to prior (Gaussian) distribution $p_1=\pi_1=\mathcal{N} (0, \mathbb{I})$.

# From end-conditioned flow to unconditional flow

**End-conditioned flow**. Given $x_0\sim \pi_0$, how to sample from the transition kernel $\mathcal{N} (x_t; \alpha_t x_0, \sigma_t^2 \mathbb{I})$? Actually, we can build a flow map to do this:

$$x_t = \alpha_t x_0 + \sigma_t \epsilon; \quad \epsilon \sim \mathcal{N} (0, \mathbb{I}). \tag{2}$$

Alternatively, we can run an ODE from $0$ to $t$:

$$dX_t = (\dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon)dt, \quad, \epsilon \sim \mathcal{N} (0, \mathbb{I}) \tag{3}$$

We denote $v_{t\mid 0}=\dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon$.

**Unconditional flow**. $\pi_1$ is a Gaussian distribution and easy to sample $x_1$ from $\pi_1$, so the problem is that how to transfer from $\pi_1$ to data distribution $\pi_0$? The answer is simple, we can run an ODE from $t=T$ to $t=0$:
$$dX_t = u_t (x_t)dt \tag{4}$$

where $v_t(x_t):=\mathbb{E}[v_{t \mid 0} \mid x_t]$.

To proof Eq. (4), we first introduce continuity equation. 

> **Theorem (Continuity Equation).**  The **continuity equation** expresses **mass (probability) conservation** of a time-evolving density $p_t(x)$ under a vector field $v_t(x)$:
> $$\frac{\partial p_t(x_t)}{\partial t} + \nabla_x \cdot (p_t v_t) = 0 \tag{5}$$

By Continuity Equation (5) and Eq. (3), $p_{t \mid 0}(x_t \mid x_0)$ satisfy:

$$\frac{\partial p_{t \mid 0}(x_t \mid x_0)}{\partial t} = - \nabla_x \cdot (p_{t \mid 0} (x_t \mid x_0) v_{t \mid 0}) \tag{6}$$

Here $\nabla_x \cdot = \text{div}(\cdot)$ is the divergence operator.

By Eq. (1), $p_t(x_t)$ satisfy:

$$\begin{align}\frac{\partial p_t(x_t)}{\partial t} &= \frac{\partial}{\partial t} \int p_t(x_t \mid x_0) \pi_0 dx_0\\ &= \int \frac{\partial}{\partial t} p_t(x_t \mid x_0) \pi_0 dx_0 \\&= \int - \nabla_x \cdot (p_{t \mid 0} v_{t \mid 0}) \pi_0 dx_0 \\&= - \nabla_x \cdot \int v_{t \mid 0} p_{t \mid 0} \pi_0 dx_0 \\&= - \nabla_x \cdot \big[p_t\int v_{t \mid 0} \frac{p_{t \mid 0}\pi_0}{p_t} dx_0   \big] \\ &=  - \nabla_x \cdot (p_t \mathbb{E}[v_t \mid x_0]) \end{align}$$
Therefore, the related ODE of $p_t$ is Eq. (4).

# Training

The conditional expectation $\mathbb{E}[f(\cdot) \mid x_t]$ is the minimizer of the following objective:

$$\mathbb{E}[f (\cdot) \mid x_t] = \arg \min \int_{0}^1 \mathbb{E}_{x_0 \sim \pi_0} \mathbb{E}_{x_t \sim p(x_t \mid x_0)} \Vert D(x_t, t) -f(\cdot)\Vert^2 dt \tag{7}$$

Eq. (7) connect the conditional expectation with a minimizer, which is the objective for training.

# Conditional expectation decomposition

The conditional expectation $\mathbb{E}[v_{t \mid 0} \mid x_t]$ can be decomposed:

$$\mathbb{E}[v_{t \mid 0} \mid x_t] = \mathbb{E}[\dot{\alpha_t}x_0 + \dot{\sigma}_t \epsilon \mid x_t] = \dot{\alpha_t} \mathbb{E}[x_0 \mid x_t] + \dot{\sigma}_t \mathbb{E}[\epsilon \mid x_t] \tag{8}$$

Since $\mathbb{E}[x_t \mid x_t]=x_t$, therefore, the following equation also holds:

$$\mathbb{E}[x_t \mid x_t] = \mathbb{E}[\alpha_t x_0 + \beta_t \epsilon \mid x_t]=\alpha_t \mathbb{E}[x_0 \mid x_t] + \sigma_t \mathbb{E}[\epsilon \mid x_t] \tag{9}$$

Eq. (8) and (9) are two equations for three terms: $\mathbb{E}[v_{t \mid 0} \mid x_t]$, $\mathbb{E}[x_0 \mid x_t]$ and $\mathbb{E}[\epsilon_t\mid x_t]$. In other words, the other two can be determined as we know one of them. This is the foundation for $x_0$-prediction, $v$-prediction and $\epsilon$-prediction. 

# Connection with the score
The score $\nabla_{x_t} \log p_t(x_t)$ is equivalent to:
$$\nabla_{x_t} \log p_t(x_t) = \mathbb{E}[\frac{\alpha_t x_0 -x_t}{\sigma_t^2} \mid x_t] = - \frac{1}{\sigma_t} \mathbb{E}[\epsilon \mid x_t] \tag{10}$$

Proof. 
For a Gaussian transition kernel $p(x_t \mid x_0) = \mathcal{N} (x_t; \alpha_t x_0, \sigma_t^2 \mathbb{I} )$, the conditional score satisfy:

$$\nabla _{x_t} p_t(x_t \vert x_0) = - \frac{x_t - \alpha_t x_0}{\sigma_t^2} p_t(x_t \vert x_0)$$

Thus,

$$ \begin{align}\nabla_{x_t} \log p_t(x_t) &= \frac{\nabla_{x_t}p_t(x_t)}{p_t}  \\ &= \frac{\nabla_{x_t} \int p_t(x_t \mid x_0) \pi_0(x_0) dx_0}{p_t} \\ &= \frac{ \int \nabla_{x_t}  p_t(x_t \mid x_0) \pi_0(x_0) dx_0}{p_t} \\ &=  \int - \frac{x_t - \alpha_t x_0}{\sigma_t^2} \frac{p_t(x_t \mid x_0) \pi_0(x_0)}{p_t}  dx_0 \\ &= \mathbb{E} [\frac{\alpha_t x_0 - x_t}{\sigma_t^2} \mid x_t]\end{align}$$

Here we connect the score with conditional expectation. Eq. (8),  (9) and (10) tell us that $\mathbb{E}[v_{t \mid 0} \mid x_t]$, $\mathbb{E}[x_0 \mid x_t]$ and $\mathbb{E}[\epsilon_t\mid x_t]$ and the score $\nabla_{x_t} p_t(x_t)$ can be determined as one of them is known. The score is useful as we want to adding stochasticity at inference time, i.e., using SDE instead of ODE at inference time.

# Stochasticity at inference time

**Lemma 1**. Consider a continuous dynamics given by ODE of the form: $dX_t = u_t dt$, with the density evolution $p_t (X_t)$. Then there exists forward SDEs and backward SDEs that match the marginal distribution $p_t$. The forward SDEs are given by: $dX_t = (u_t +  \frac12 \zeta_t^2 \nabla_{x_t}\log p_t )dt + \zeta_t d W_t$, where $W_t$ is Wiener process. The backward SDEs are given by: $dX_t = (u_t - \frac12 \zeta_t^2 \nabla_{x_t} \log p_t) dt + \zeta_t d W_t$.

Lemma 1 tell us that there exists a series of SDEs that share the same marginal of an ODE. Therefore, we can extend ODE in Eq. (4) to SDEs:
$$dX_t = (v_t(x_t)- \frac12 \zeta_t^2 \nabla_{x_t} \log p_t)dt + \zeta_t dW_t \tag{11}$$
Note that $v_t(x_t)$ and $\nabla_{x_t} \log p_t(x_t)$ can be determined as one of them is known.

