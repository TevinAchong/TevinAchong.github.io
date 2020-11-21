---
layout: post
title: Notes on Training Machine Learning Models
published: true
---

This post contains my notes about training machine learning models. The majority of content is information summarized from Aurelien Geron's book [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646/ref=pd_lpo_14_t_0/139-2845639-1288614?_encoding=UTF8&pd_rd_i=1492032646&pd_rd_r=b7d9fcc6-fb0b-4355-9914-8946ce444f49&pd_rd_w=wr8Wm&pd_rd_wg=2Ltb3&pf_rd_p=7b36d496-f366-4631-94d3-61b87b52511b&pf_rd_r=HEY5HJ6F40FFSND2R3BW&psc=1&refRID=HEY5HJ6F40FFSND2R3BW).

### The Linear Regression Model

Two different ways to train the linear regression model are discussed:
- Using a direct "closed-form" equation that directly computes the model parameters that best fit the model to the training set (those parameter values that minimize the cost function over the training set).
- Using Gradient Descent (GD), and some of its variants Batch GD, Mini-Batch GD and Stochastic GD. GD is an iterative optimization approach that gradually tweaks the model parameters to minimize the cost function over the training set, eventually converging to the same set of parameters as the first method. 

A linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the *bias term*. 

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

Where:
- $\hat{y}$ is the predicted value
- $n$ is the number of features
- $x_i$ is the $i^{th}$ feature value
- $\theta_j$ is the $j^{th}$ model parameter (including the bias term $\theta_0$ and the feature weights $\theta_1, \theta_2, ..., \theta_n$). 

We can express this more concisely using the vectorized form:

$$
\hat{y} = h_0(x) = \theta \dot x
$$

Where:
- $\theta$ is the model's *parameter vector*, containing the bias term $\theta_0$ and the feature weights $\theta_1$ to $\theta_n$. 
- $x$ is the instance's *feature vector*, containing $x_0$ to $x_n$, with $x_0$ always equal to 1. 
- $\theta \dot x$ is the dot product of the vectors $\theta$ and $x$ ($\theta_0x_0 + \theta_1x_1 + ... + \theta_nx_n$).
- $h_0$ is the hypothesis function, using the model parameters $\theta$.
