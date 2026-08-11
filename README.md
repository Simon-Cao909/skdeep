# skdeep
skdeep is a scikit-learn compatible deep learning framework built on top of Keras that lets one describe complex architectures as composable graph blocks. It is also fully compatible with scikit-learn's estimator API

## Features

Graph-based architecture DSL

Estimators, regressors, and classifiers
- Supports modern architectures (ResNet, Inception, Xception, etc.)

Standard and variational autoencoders

Physics Informed Neural Networks (PINN) and Inverse PINNs
- Supports easy-declaration (ex. "u_tt - (c^2)u_xx = 0" is all you need to write for the equation!)
- Supports vector calculus (Maxwell's equations, Laplace equation, Navier-Stokes, etc.)
- Think of Desmos but in Python and for PDEs

scikit-learn BaseEstimator compatibility

## Getting started

### pip installation

```bash
python -m pip install -U skdeep
```

### From source

```bash
git clone https://github.com/Simon-Cao909/skdeep.git
cd skdeep
python -m pip install .
```

## Documentation

You can find documentation on https://skdeep.readthedocs.io/en/latest/

You can also find some important documents in docs/ (particularly architecture.md and equation.md)

## Examples

To see more examples, look at examples/

Regression:

```python
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers as kl

from skdeep.regressor import DeepRegressor
from generate_data import get_multi_input_data

X1,X2,y = get_multi_input_data()

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1,X2,y,train_size=0.8,random_state=42)

model = DeepRegressor(model_structure=[
        ['multi-input',[
            [['dense',64,'relu']],
            [['dense',64,'relu']]
        ],kl.Add()],
        ['dense',64,'relu'],
        ['dropout',0.1],
        ['dense',32,'relu'],
        ['dense',1,'linear'],
    ],
    build_setting="quick",
    input_shape=[(4,),(5,)],
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    random_state=42,
    loss='mse'
)

model.fit([X1_train,X2_train],y)
```

Classification:

```python
from skdeep.classifier import DeepClassifier
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32').reshape(x_train.shape[0],28,28,1) / 255
x_test = x_test.astype('float32').reshape(x_test.shape[0],28,28,1) / 255

model = DeepClassifier(model_structure = [
                                                {'type':'C', 'filters':8, 'kernel_size':(3,3), 'activation':'relu', 'padding':'same'},
                                                {'type':'MP'},

                                                {'type':'R',
                                                 'layers':[
                                                    {'type':'C', 'filters':8, 'kernel_size':(3,3), 'activation':'relu', 'padding':'same'},
                                                    {'type':'C', 'filters':16, 'kernel_size':(3,3), 'activation':'relu', 'padding':'same'},
                                                 ],
                                                 'final_activation':'linear',
                                                 'allow_projection':True
                                                },
                                                
                                                {'type':'MP'},

                                                {'type':'F'},
                                                {'type':'D', 'units':32, 'activation':'relu'},
                                                {'type':'D', 'units':10, 'activation':'softmax'}
                                            ],
                          epochs = 3,
                          learning_rate = 1e-3,
                          loss = 'categorical_crossentropy',
                          optimizer = 'adam',
                          batch_size = 128,
                          verbose = 1,)

model.fit(x_train,y_train)

print("Accuracy score:", model.score(x_test,y_test))
```

Autoencoder:

```python
from skdeep.autoencoder import DeepAutoencoder
from tensorflow import keras
import numpy as np

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32').reshape(x_train.shape[0],28,28,1) / 255
x_test = x_test.astype('float32').reshape(x_test.shape[0],28,28,1) / 255

x_train = x_train[:10000]
x_test = x_test[:5000]

model = DeepAutoencoder(encoder_structure=[
                                    {'type':'C','filters':16,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'MP','pool_size':(2,2),'strides':(2,2)},
                                    {'type':'C','filters':32,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'F'},
                                    {'type':'D','units':100,'activation':'linear'}
                            ],
                            decoder_structure=[
                                    {'type':'D','units':7*7*32,'activation':'relu'},
                                    {'type':'custom','layer':keras.layers.Reshape((7,7,32))},
                                    {'type':'CT','filters':32,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'UP','size':(2,2)},
                                    {'type':'CT','filters':16,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'UP','size':(2,2)},
                                    {'type':'CT','filters':1,'kernel_size':(3,3),'activation':'sigmoid','padding':'same'},
                            ],
                            model_type='standard', # or variational
                            epochs=3,
                            batch_size=128,
                            verbose=1,
                            optimizer='adam',
)

model.fit(x_train)
pred = model.predict(x_test)
```

PINNs:

```python
from skdeep import DeepPINN
import numpy as np
import tensorflow.keras.ops as ko

variables = ["x", "y", "z", "t"]
functions = ["E", "B"]

constants = [
    {
        "name": "ε", "value": 1,
        "trainable": False,
    },
    {
        "name": "μ", "value": 1,
        "trainable": False,
    }
]

equation_structure = [
    "∇[x,y,z]⋅E = 0",
    "∇[x,y,z]⋅B = 0",
    "∇[x,y,z]×E = -B_t",
    "∇[x,y,z]×B = E_t",
]

conditions = [

    # E(x,y,z,0) = [sin(y), 0, 0]
    {
        "location": {"t": 0}, "n_samples": 200, "equation": "E1 - ()sin(y) = 0"
    },

    # E2(x,y,z,0) = 0
    {
        "location": {"t": 0}, "n_samples": 200, "equation": "E2 = 0"
    },

    # E3(x,y,z,0) = 0
    {
        "location": {"t": 0}, "n_samples": 200, "equation": "E3 = 0"
    },

    # B1(x,y,z,0) = 0
    {
        "location": {"t": 0}, "n_samples": 200, "equation": "B1 = 0"
    },

    # B2(x,y,z,0) = 0
    {
        "location": {"t": 0}, "n_samples": 200, "equation": "B2 = 0"
    },

    # B3(x,y,z,0) = -sin(y)
    {
        "location": {"t": 0}, "n_samples": 200, "equation": "B3 = -()sin(y)"
    }
]

bounds = {"x": (0, 2 * np.pi), "y": (0, 2 * np.pi), "z": (0, 2 * np.pi), "t": (0, 1)}

model_structure = [
    ["Norm",np.asarray([0, 0, 0, 0]),np.asarray([2*np.pi, 2*np.pi, 2*np.pi, 1])],
    ["Dense", 96, "tanh"],
    ["Dense", 64, "tanh"],
    [
        "multi-output",
        [
            [["Dense", 3, "linear"]],  # E
            [["Dense", 3, "linear"]]   # B
        ]
    ]
]

pinn = DeepPINN(
    variables=variables,
    equation_structure=equation_structure,
    conditions=conditions,
    bounds=bounds,
    n_samples=500,
    constants=constants,
    functions=functions,
    loss_weighting={
        "pde": 1.0,
        "conditions": 2.0
    },
    model_structure=model_structure,
    build_setting="quick",
    epochs=200,
    early_stopping=True,
    optimizer="adam"
)

pinn.fit()
```