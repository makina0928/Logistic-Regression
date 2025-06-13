# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # LOGISTIC REGRESSION AS A NEURAL NETWORK

    ## Binary classification

    In binary classification, a goal is to learn a classifier than can input an image represented by its feature vector $X$ and predict weither the corresponding label $Y$ is $1$ or $0$.<br> If we let a single training example be represented by a pair $(x, y)$, where $x \in \mathbb{R}^{n_x}$ ($x$ is in $n_x$ dimensional feature vector) and $y$ label is either $0$ or $1$ given by $y \in \{0, 1\}$ then:

    The input and output of each training example is given by:
    $$m_{train} = (x^{(1)},y^{1}), (x^{(2)},y^{2}),\dots, (x^{(n)},y^{n})$$

    In more compact notation:

    $$
    X = 
    \begin{bmatrix}
    x^{(1)} & x^{(2)} & \cdots & x^{(m)} 
    \end{bmatrix}
    \in \mathbb{R}^{n_x \times m}
    $$

    Where: $m$ is the number of training samples.

    For $Y$: 

    $$
    Y = \begin{bmatrix}
    y^{(1)} & y^{(2)} & \cdots & y^{(m)}
    \end{bmatrix}\in \mathbb{R}^{1 \times m}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Logistic Regression (LR) Model

    In binary classification tasks such as determining whether an image belongs to a dog or not, we are given an input feature vector $x \in \mathbb{R}^{n_x}$. Our goal is to predict a binary label $y \in \{0, 1\}$, where:

    * $y = 1$ might represent "dog"
    * $y = 0$ might represent "not a dog"

    We want to estimate the probability that $y = 1$ given the input $x$. This estimate is denoted as:

    $$
    \hat{y} = P(y = 1 \mid x)
    $$

    ### Model Parameters

    The logistic regression model has two trainable parameters:

    $$
    w \in \mathbb{R}^{n_x}, \quad b \in \mathbb{R}
    $$

    Where:<br>

    * $w$: weight vector
    * $b$: bias term

    ### Model Output

    The model computes a linear combination of inputs followed by a nonlinear activation using the sigmoid function:

    $$
    \hat{y} = \sigma(z), \quad \text{where } z = w^\top x + b
    $$

    The sigmoid function transforms the linear output $z$ into a value between $0$ and $1$, so that it can be interpreted as a probability.


    The sigmoid function is defined as:

    $$
    \sigma(z) = \frac{1}{1 + e^{-z}}
    $$

    #### The behavior of the function is such that:

    * If $z$ is a large positive number, then:

    $$
    \sigma(z) \approx \frac{1}{1 + 0} = 1
    $$

      Which means the model is very confident that $y = 1$

    * If $z$ is a large negative number, then:

    $$
    \sigma(z) \approx \frac{1}{1 + \text{big number}} \approx 0
    $$

      Meaning the model is very confident that $y = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loss (error) function

    Once we’ve defined our model and how it makes predictions, the next step is to measure how good those predictions are. This is done using a **loss function**.

    We aim to learn the optimal values for the parameters $w$ and $b$ such that our model's predictions are as close as possible to the true labels in the training dataset.

    ### Objective

    Given a training set of $m$ labeled examples:

    $$
    \{(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})\}
    $$

    we want the predicted output $\hat{y}^{(i)}$ to be close to the actual label $y^{(i)}$ for each training example $i$. That is,

    $$
    \hat{y}^{(i)} \approx y^{(i)}
    $$


    For each training example $i$, the model predicts:

    $$
    \hat{y}^{(i)} = \sigma(w^\top x^{(i)} + b), \quad \text{where } \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}
    $$

    * $\sigma(\cdot)$ is the sigmoid function, which maps real-valued inputs to the range $(0, 1)$.
    * The output $\hat{y}^{(i)}$ is interpreted as the probability that $y^{(i)} = 1$ given input $x^{(i)}$.

    The loss function $L(\hat{y}, y)$ (binary cross-entropy loss) tells us how well the predicted output $\hat{y}$ matches the true label $y$ and is given by: 

    $$
    \mathcal{L}(\hat{y}, y) = - \left( y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \right)
    $$

    ### Loss Function Behavior for Each Class

    - If \( y = 1 \):

    $$
    \mathcal{L}(\hat{y}, y) = -\log \hat{y}
    $$

      * To minimize the loss, we want \( \log \hat{y} \) to be large  
      * That means we want \( \hat{y} \) to be **close to 1**

    - If \( y = 0 \):

    $$
    \mathcal{L}(\hat{y}, y) = -\log (1 - \hat{y})
    $$

      * To minimize the loss, we want \( \log(1 - \hat{y}) \) to be large  
      * That means we want \( \hat{y} \) to be **close to 0**


    To give a scalar value that represents how well the model is performing on the entire dataset, the *Cost function* averages the binary cross-entropy loss over all training examples and is given by: 

    $$
    \begin{align*}
    \text{Cost function:} \quad J(w, b) &= \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})\\ &= -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
    \end{align*}
    $$

    **Note:** The loss function is applied to the single training example, while the cost function is the cost of our parameters. Therefore, in training our logistic regression model, we try to find the parameters that minimizes the *Cost function*
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Gradient Descent Algorithm for Learning Parameters 

    The aim here is to find **$w$** and **$b$** that minimize **$J(w,b)$**.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
