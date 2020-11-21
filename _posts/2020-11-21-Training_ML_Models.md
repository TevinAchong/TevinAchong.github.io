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

$$ \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n $$
