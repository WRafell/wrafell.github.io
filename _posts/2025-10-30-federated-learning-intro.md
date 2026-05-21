---
title: "Introduction to Federated Learning"
date: 2025-10-30
categories:
  - notes
tags:
  - federated-learning
  - machine-learning
excerpt: "A decentralized ML paradigm where clients collaboratively train a model without sharing private data."
mathjax: true
---
# Introduction to Federated Learning

## Summary

Federated Learning (FL) is a decentralized machine learning paradigm where multiple clients (e.g., mobile devices, organizations, IoT) collaboratively train a model coordinated by one or more central servers, without sharing private data. It addresses challenges like data privacy and data silos — isolated data accessible only to specific groups. Clients download a global model, train it locally on private data, and send encrypted updates (gradients or weights) to the server. The server aggregates these to update the global model, repeating until convergence. Data heterogeneity (non-IID data) often complicates convergence.

## Why It Matters

FL enables collective intelligence without compromising user data privacy or requiring data centralization. It provides a practical solution for training models across distributed and sensitive datasets.

## Context

The concept was popularized by Google researchers (McMahan et al., 2017) through the introduction of FedAvg, a baseline algorithm used to train text prediction models across thousands of Android devices while keeping data on-device.

## Standard Process

FL typically follows an *aggregate-then-adapt* loop:

1. **Clients** download the global model and train it on private data.
2. Locally updated gradients or weights are **uploaded** to the central server, often with encryption.
3. The server **aggregates** updates to form a new global model.
4. The new model is redistributed, and the cycle continues until convergence.

## Key Challenges

- **Data Heterogeneity (non-IID data):** Clients often have diverse, non-identically distributed data, which can slow or destabilize convergence.
- **Communication Efficiency:** Frequent model exchange can be bandwidth-intensive.
- **Privacy and Security:** Requires secure aggregation and client selection mechanisms.

---

## FedAvg — Mathematical Formulation

### Problem Setup

Consider $K$ clients, where client $k$ holds a local dataset $\mathcal{D}_k$ of size $n_k$. The total number of samples is $n = \sum_{k=1}^K n_k$. Each client defines a local objective:

$$F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)$$

where $\ell$ is a per-sample loss. The global objective is a weighted average of local objectives:

$$\min_w \; f(w) = \sum_{k=1}^K \frac{n_k}{n} F_k(w)$$

### The FedAvg Algorithm

Each communication round $t$ proceeds as follows:

**1. Client selection.** The server samples a subset $S_t \subseteq [K]$ of clients (fraction $C$).

**2. Local update.** Each selected client $k$ initializes from the global model $w^t$ and runs $E$ epochs of SGD with batch size $B$:

$$w_k^{t,0} = w^t$$

$$w_k^{t,j+1} = w_k^{t,j} - \eta \, \nabla \ell(w_k^{t,j};\, \xi_j^k)$$

After $E$ epochs, the client holds a locally updated model $w_k^{t,E}$.

**3. Aggregation.** The server forms the new global model as a weighted average:

$$w^{t+1} = \sum_{k \in S_t} \frac{n_k}{n_t} \, w_k^{t,E}$$

where $n_t = \sum_{k \in S_t} n_k$.

### Special Case — FedSGD

Setting $E = 1$ and $B = n_k$ (full-batch local gradient) recovers **FedSGD**, which is equivalent to centralized gradient descent:

$$w^{t+1} = w^t - \eta \sum_{k=1}^K \frac{n_k}{n} \nabla F_k(w^t)$$

FedAvg generalizes this by allowing multiple local steps before aggregation, reducing communication rounds at the cost of potential client drift.

### Key Hyperparameters

| Symbol | Meaning |
|--------|---------|
| $K$ | Total number of clients |
| $C$ | Fraction of clients selected per round |
| $E$ | Number of local training epochs |
| $B$ | Local minibatch size |
| $\eta$ | Local learning rate |

### Client Drift

With non-IID data, local distributions $\mathcal{D}_k$ differ across clients. Each local model drifts toward its own optimum $w_k^\* = \arg\min F_k(w)$, which may be far from the global optimum $w^\* = \arg\min f(w)$. This causes the aggregated model to diverge from the true global minimum — a phenomenon known as **client drift**. Larger $E$ amplifies drift.

Formally, the bias introduced is related to the gradient divergence:

$$\mathbb{E}\left[\|\nabla F_k(w) - \nabla f(w)\|^2\right]$$

which is non-zero when data is heterogeneous.

---

**Source:** McMahan et al., 2017 — *Communication-Efficient Learning of Deep Networks from Decentralized Data*
