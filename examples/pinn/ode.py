import numpy as np
import tensorflow as tf
import tensorflow.keras.ops as ko

from skdeep.pinn import DeepPINN


# ----------------------------------
# Model architecture
# ----------------------------------

model_structure = [
    {
        "type": "Norm",
        "mins": np.array([0]),
        "maxs": np.array([1]),
    },
    {
        "type": "Dense",
        "units": 32,
        "activation": "tanh"
    },
    {
        "type": "Dense",
        "units": 32,
        "activation": "tanh"
    },
    {
        "type": "Dense",
        "units": 1,
        "activation": "linear"
    }
]


# ----------------------------------
# ODE:
#
# u_xx + pi^2 sin(pi*x)=0
#
# solution u=sin(pi*x)
#
# ----------------------------------

equation_structure = [
    {
        "var": "u",
        "derivatives": ["x","x"],
        "coef": 1
    },
    {
        "var": "x",
        "op": lambda x: np.pi**2 * ko.sin(np.pi * x),
        "coef": 1
    }
]


# ----------------------------------
# Boundary:
#
# u(0)=0
# u(1)=0
#
# ----------------------------------

conditions = [
    {
        "location":{"x":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },
    {
        "location":{"x":1},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    }
]


# ----------------------------------
# Create estimator
# ----------------------------------

pinn = DeepPINN(
    variables=["x"],
    equation_structure=equation_structure,
    conditions=conditions,
    bounds={
        "x":(0,1)
    },
    n_samples=1000,


    # inherited DeepEstimator params
    model_structure=model_structure,
    epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    validation_split=0.1,
    early_stopping=True,
    verbose=1,
    random_state=42
)


# ----------------------------------
# Fit
# ----------------------------------

pinn.fit()


# ----------------------------------
# Check model
# ----------------------------------

x = np.linspace(
    0,
    1,
    200
).reshape(-1,1)


prediction = pinn.predict(x)

truth = np.sin(np.pi*x).ravel()


print(
    "solution MSE:",
    np.mean((prediction-truth)**2)
)

import matplotlib.pyplot as plt
print(x.shape)
print(prediction.shape)
plt.plot(x,prediction,label="Pred",color='red')
plt.plot(x,truth,label="Truth",color='black')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()
plt.close()

# ----------------------------------
# Check PDE residual
# ----------------------------------

residual = pinn._calc_pde(
    tf.constant(
        x,
        dtype=tf.float32
    )
)


print(
    "PDE residual:",
    tf.reduce_mean(
        tf.abs(residual)
    ).numpy()
)


# ----------------------------------
# Check sklearn stuff
# ----------------------------------

print("Estimator params:")
print(pinn.get_params())

print("Score:")
print(pinn.score(x))