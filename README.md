# skdeep
skdeep is a graph-based deep learning framework built on top of Keras that lets one describe complex architectures as composable graph blocks. It is also fully compatible with scikit-learn's estimator API

## Features

Graph-based architecture DSL

ResNet, Inception, and Xception

Transfer learning

Embed pre-trained scikit-learn regressors into neural networks

Standard and variational autoencoders

Physics Informed Neural Networks (PINN) and Inverse PINNs
- Supports vector calculus (Maxwell's equations, Laplace equation, Navier-Stokes, etc.)

scikit-learn BaseEstimator compatibility

## Getting started

```bash
pip install skdeep
```

```python
from skdeep.estimator import DeepEstimator
```

## Documentation

You can find documentation on https://skdeep.readthedocs.io/en/latest/

## Why I built this
Often, when performing research, it's useful to test different machine learning models. Whether that be through tweaking the hyperparameters or changing your framework, it was interesting to see how different models performed.

However, when trying to work with neural networks, I encountered a block. I either had to use sklearn's MLPRegressor, which significantly limits your ability, or attempt to work with KerasRegressor from scikeras, which still requires you to rewrite dozens of lines of code when attempting to use a different model architecture. Debugging was a chore, and for more complex networks like Inception, ResNet, and Xception, it was a pain to code and read.

This is why I created skdeep. It allows you to easily specify complex model structures with something resembling a flow chart, making it significantly more user-friendly while maintaining most of Keras's versatility. You can create complex and more readable just by coding something like:
```python
[
    {'type':'C', 'filters':32,
     'kernel_size':(3,3),
     'padding':'same',
     'activation':'relu'},

    {
        'type':'R',
        'layers':[
            {'type':'C','filters':32,
             'kernel_size':(3,3),
             'padding':'same',
             'activation':'relu'},

            {'type':'C','filters':32,
             'kernel_size':(3,3),
             'padding':'same',
             'activation':'linear'}
        ],
        'final_activation':'relu'
    },

    {'type':'GAP'},
    {'type':'D','units':1,'activation':'linear'}
]
```
Then, with many model structures created already, you can run it through a GridSearchCV or add it to your code alongside any other sklearn regressor, and it will behave just fine.