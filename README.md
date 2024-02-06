# Linear Regression with Numpy
<br><br>
## Overview<br>
This repository contains a Python implementation of Linear Regression using only the Numpy library. Linear Regression is a fundamental algorithm in machine learning, and this implementation serves as a minimalistic yet effective example using numpy for educational purposes.
<br><br>
## Purpose<br>
Multi-Linear Regression allows for modeling relationships between multiple independent variables and a single dependent variable. In this context:
<br><br>
X: Represents the input matrix with multiple features.<br>
Y: Represents the target variable we aim to predict.<br>
Gradient Descent: Optimizes the model to find the best-fitting parameters (theta) for accurate predictions.<br>
This implementation serves as an educational resource to understand the inner workings of multi-linear regression and the implementation nuances using only the Numpy library.
<br><br>
The primary purpose of this implementation is to showcase the core concepts of Linear Regression, including hypothesis formulation, cost computation, gradient descent optimization, and model training. The code emphasizes simplicity and clarity to aid understanding.
<br><br>
## Code Highlights <br>
### Linear Regression Class
<br><br>
![Formula](https://github.com/hisanusman/Linear-Regression-through-Numpy/assets/101946933/a141c149-1095-4e37-9151-e34eb2fb9acd)

The `LinearRegression` class encapsulates the entire process of linear regression. Here's a breakdown of its key components:
<br><br>
- **Hypothesis Function**: Computes the linear hypothesis with an added bias term to the input matrix.
- **Cost Function**: Calculates the mean squared error between predicted values and actual labels.
- **Derivative Function**: Computes the derivative of the cost function with respect to the model parameters (theta).
- **Gradient Descent**: Updates model parameters using the calculated derivative and a specified learning rate.
- **Training**: Initializes random theta values and iteratively performs gradient descent to optimize the model.
- **Prediction**: Uses the trained model to make predictions on new data.
- **Get Weights**: Returns the flattened theta values.
<br><br>
### Code Structure <br>
The code is structured to be modular and comprehensible. Each function is well-defined with comments to facilitate understanding. The structure is designed to encourage exploration and modification for learning purposes.
<br><br>
## Usage <br>
To use the linear regression model:

```python
# Example Usage

# Initialize the model
model = LinearRegression()

# Train the model
model.train(X, Y, lr=0.0001, num_iters=1000, num_printing=100)

# Make predictions
predictions = model.predict(X_test)

# Get model weights
weights = model.get_weights()
```
<br><br>
Feel free to reach out for any questions, suggestions, or improvements!
