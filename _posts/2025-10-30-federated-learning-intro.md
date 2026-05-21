---
title: "Introduction to Federated Learning"
date: 2025-10-30
categories:
  - notes
tags:
  - federated-learning
  - machine-learning
excerpt: "A decentralized ML paradigm where clients collaboratively train a model without sharing private data."
---

**Source:** McMahan et al., 2017 — *Communication-Efficient Learning of Deep Networks from Decentralized Data*

---

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
