
# Jedi Machine Learning Suite
- A self-implemented machine learning algorithm suite, inspired by Professor Hsuan-Tien Lin, National Taiwan University, [machine learning course](https://www.youtube.com/user/hsuantien/playlists).
- The suite aims to help beginners to understand theory step-by-step behind foundational machine learning algorithm, including Perceptron, Regression, Support Vector Machine, Decision Tree etc.

Installation
============
``` shell
$ pip install JediML
```
- Note : The suite is still under development. Please check back if error occurs. Will fix it as soon as possible!

Algorithm
============
- Perceptron Learning
    - Perceptron Learning Algorithm Binary/Multi Classification 
    - Pocket Perceptron Learning Algorithm Binary/Multi Classification 
- Regression
    - Linear Regression Learning 
    - Ridge Regression Learning 
    - Kernel Ridge Regression Learning
- Logistic Regression
    - Logistic Regression Learning
    - L2 Regularized Logistic Regression Learning
    - Kernel Logistic Regression Learning
- Support Vector Machine
    - Hard Margin SVM
    - Polynomial Kernel SVM
    - Gaussian Kernel SVM
    - Polynomial/ Soft Polynomial Kernel SVM
    - Gaussian / Soft Gaussian Kernel SVM
    - Probabilistic SVM
    - Least Squares SVM
    - Support Vector Regression SVM
- Decision Tree
    - Decision Stump Based
    - AdaBoost Stump Based
    - AdaBoost Decision Tree
    - Gradient Boost Decision Tree
    - Decision Tree Classification/Regression
    - Random Forest Classification/Regression
- Neural Network
    - Neural Network Binary Classification
- Accelerator
    - Linear Regression Accelerator
- Feature Transform
    - Polynomial Feature/Legendre Feature Transform
- Validation
    - N Fold Cross Validation
- Blending
    - Uniform Blending for Classification/Regression
    - Linear Blending for Classification/Regression

Usage
============
```python
    >>> import numpy as np
    >>> import JediML.PLA as pla

    >>> your_input_data_file = '/path/to/your/data/file'
    # Suite also provide sample data from Professor Hsuan-Tien Lin course material

    >>> pla_bc = pla.BinaryClassifier()
    >>> pla_bc.load_train_data(your_input_data_file)
    >>> pla_bc.set_param()
    >>> pla_bc.init_W()
    >>> W = pla_bc.train()

    >>> test_data = 'Format : Each feature of data x separated with spaces, and the ground truth y at the end of line.'
    # assign test data, format like this '0.97681 0.10723 0.64385 ........ 0.29556 1'

    >>> prediction = pla_bc.prediction(test_data)

    >>> print prediction['input_data_x']
    >>> print prediction['input_data_y']
    >>> print prediction['prediction']
```

PEP8
=========

```shell
   pep8 JediML/*.py --ignore=E501
```
License
=========
The MIT License (MIT)
