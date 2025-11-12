---
layout: post
title: "Learn Conditional Expectations via Bregman divergence"
date: 2025-11-11
tags: [reading notes]
---

# Bregman divergence

The **Bregman divergence** measures how far two points are from each other **with respect to a convex function** — it generalizes familiar notions like squared Euclidean distance and Kullback–Leibler (KL) divergence.

## Definition
Let $\phi: \mathbb{R}^n \rightarrow \mathbb{R}$ be a strictly convex, differentiable function. The Bregman divergence between two points $x$ and $y$ is defined as:

$$D_{\phi} (x, y) = \phi(x)-\phi(y) - \langle\nabla \phi(y), x-y \rangle$$

where $\phi(x)$ is value of the convex function at $x$, $\nabla \phi(y)$ is gradient at $y$, $\langle \cdot, \cdot \rangle$ denotes inner product. Intuitively, $D_{\phi}(x, y)$ measures how much $\phi(x)$ exceeds the tangent plane to $\phi$ at $y$.
## Examples
**Squared Euclidean Distance**. If $\phi(x)=\frac{1}{2}\Vert x \Vert^2$, then
$$D_{\phi}(x, y)=\frac12 \Vert x-y \Vert_2^2$$
So the Euclidean distance squared is a Bregman divergence.
**Kullback–Leibler (KL) Divergence**. If $\phi(p)=\sum_{i}p_i\log p_i$ (the negative entropy), then 

$$D_{\phi}(p, q)=\sum_{i}p_i \log \frac{p_i}{q_i}$$
That’s the KL divergence between distributions $p$ and $q$.


# Learn Conditional Expectations via Bregman divergence


> Proposition 1 [1].  Let $X\in \mathcal{S}_X, Y \in \mathcal{S}_Y$ be RVs over state spaces $\mathcal{S}_X$, $\mathcal{S}_Y$ and $g: \mathbb{R}^p \times \mathcal{S}_X \rightarrow \mathbb{R}^n$, $(\theta, x) \mapsto g^\theta(x)$, where $\theta \in \mathbb{R}^p$ denotes learnable parameters. Let $D_X(u, v),\, x \in \mathcal{S}_X$ be a Bregman divergence over a convex set $\Omega \subset \mathbb{R}^n$ that contains the image of $f$. 
> Then,
> $$\nabla_\theta \mathbb{E}_{X,Y}\, D_X\!\big(Y, g^\theta(X)\big)
= \nabla_\theta \mathbb{E}_X\, D_X\!\big(\mathbb{E}[Y \mid X], g^\theta(X)\big).
$$
In particular, for all $x$ with $p_X(x) > 0$, the global minimum of $g^\theta(x)$ with respect to $\theta$ satisfies
> $$g^\theta(x) = \mathbb{E}[Y \mid X = x].
$$



## Learn conditional expectation in diffusion model

 For continuous diffusion models, $\mathbb{E}[f(x_0) \mid x_t]:= \int f(x_0) p( x_0\mid x_t) dx_0$. The conditional expectation $\mathbb{E}[f(x_0) \mid x_t]$ is the minimizer of the following objective:

  
$$\mathbb{E}[f (x_0) \mid x_t] = \arg \min_{g(x_t, t)} \ \mathbb{E}_{x_0 \sim \pi_0} \mathbb{E}_{x_t \sim p(x_t \mid x_0)} \Vert g(x_t, t) -f(x_0)\Vert^2 \tag{1}$$


Eq. (1) connect the conditional expectation with a minimizer, which is the objective for training.

For discrete diffusion models, we want to learn

$$p_{0 \mid t}^{i}(x_0^i \mid x_t)= \mathbb{E}[\delta (X_0^i=x_0^i) \mid X_t =x_t]$$

Here we use KL-divergence instead of Squared Euclidean Distance. The loss is:

$$\mathbb{E}_{x_0 \sim \pi_0, x_t \sim p(x_t \mid x_0)} D \left(\delta (X_0^i=x_0^i), p_{0 \mid t}^{\theta, i}(X_0^i=x_0^i \mid X_t= x_t) \right)$$

where $D(p, q)= \sum_{\alpha \in \mathcal{\tau}} p(\alpha) \log \frac{p(\alpha)}{q(\alpha)}$. Since minimize KL divergence is equivalent to maximize likelihood, we have 

$$ \arg \min - \mathbb{E}_{x_0 \sim \pi_0, x_t \sim p(x_t \mid x_0)} \log p_{0 \mid t}^{\theta, i} (X_0^i=x_0^i \mid X_t=x_t)$$


[1] Lipman Y, Havasi M, Holderrieth P, et al. Flow matching guide and code[J]. arXiv preprint arXiv:2412.06264, 2024.