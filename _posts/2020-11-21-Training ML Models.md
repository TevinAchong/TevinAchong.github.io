---
layout: post
title: Notes on Training Linear Regression Models
published: true
---

This post contains my notes about training simple linear regression models.

## The Linear Regression Model

Two different ways to train the linear regression model are discussed:
- Using a direct "closed-form" equation that directly computes the model parameters that best fit the model to the training set (those parameter values that minimize the cost function over the training set).
- Using Gradient Descent (GD), and some of its variants Batch GD, Mini-Batch GD and Stochastic GD. GD is an iterative optimization approach that gradually tweaks the model parameters to minimize the cost function over the training set, eventually converging to the same set of parameters as the first method. 

A linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the *bias term*. 

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

Where:
- $ \hat{y} $ is the predicted value
- $ n $ is the number of features
- $ x_i $ is the $ i^{th} $ feature value
- $ \theta_j $ is the $ j^{th} $ model parameter (including the bias term $ \theta_0 $ and the feature weights $ \theta_1, \theta_2, ..., \theta_n $). 

We can express this more concisely using the vectorized form:

$$
\hat{y} = h_\theta(x) = \theta \cdot x
$$

Where:
- $ \theta $ is the model's *parameter vector*, containing the bias term $ \theta_0 $ and the feature weights $ \theta_1 $ to $ \theta_n $. 
- $ x $ is the instance's *feature vector*, containing $ x_0 $ to $ x_n $, with $ x_0 $ always equal to 1. 
- $ \theta \cdot x $ is the dot product of the vectors $ \theta $ and $ x $ ($ \theta_0x_0 + \theta_1x_1 + ... + \theta_nx_n $).
- $ h_\theta $ is the hypothesis function, using the model parameters $ \theta $.

## How Do We Train the Linear Regression Model
Recall that training a model means setting its parameters so that the model best fits the training sets. 

We need a measure of how well (or poorly) the model fits the training data. The most common performance measure of a regression model is the **Root Mean Squared Error (RMSE)**:

$$
\text{RMSE}(X, h) = \sqrt{\frac{1}{m} \sum^m_{i=1} (h(x^{(i)}) - y^{(i)})^2}
$$

To train a Linear Regression model, we need to find the value of $\theta$ that minimizes the RMSE. 

In practice, it is simpler to minimize the **MSE** than the RMSE and it leads to the same result (because the value that minimizes a function also minimizes its square root).

$$
\text{MSE}(X, h_\theta) = \frac{1}{m}\sum^m_{i=1}(\theta^Tx^{(i)}-y^{(i)})^2
$$

The above equation is the MSE of a Linear Regression hypothesis $h_\theta$ on a training set $X$. 

For simplification, we represent this as:

$$
MSE(\theta)
$$

to stress that the model is parametrized by the vector $\theta$.

### The Normal Equation
This is a mathematical equation that directly gives the value of $\theta$ that minimizes the cost function - a *closed form solution*. 

$$
\hat{\theta} = (X^TX)^{-1}X^Ty
$$

Where:
- $\hat{\theta}$ is the value of $\theta$ that minimizes the cost function
- $y$ is the vector of target values containing containing $y^{(1)}$ to $y^{(m)}$

Performing Linear Regression with Scikit-Learn is relatively simple:

```
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.predict(X_new)
```

The `LinearRegression` class is based on the `scipy.linalg.lstsq()` function (which stands for "least squares"). 










