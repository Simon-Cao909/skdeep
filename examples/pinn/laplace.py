import numpy as np
import tensorflow as tf
from tensorflow.keras import ops as ko

from skdeep.pinn import DeepPINN

# ----------------------------------
# Model architecture
# ----------------------------------

model_structure = [
    {
        "type": "Norm",
        "mins": np.array([0,0]),
        "maxs": np.array([1,1]),
    },
    {
        "type": "Dense",
        "units": 64,
        "activation": "tanh"
    },
    {
        "type": "Dense",
        "units": 64,
        "activation": "tanh"
    },
    {
        "type": "Dense",
        "units": 64,
        "activation": "tanh"
    },
    {
        "type": "Dense",
        "units": 1,
        "activation": "linear"
    }
]


# ----------------------------------
# PDE:
#
# u_xx + u_yy = 0
#
# Solution:
#
# u(x,y)=sinh(pi*x)*sin(pi*y)
#
# ----------------------------------

equation_structure = "u_xx + u_yy"


# ----------------------------------
# Boundary conditions
# ----------------------------------

conditions = [

    # x = 0
    # u(0,y)=0
    {
        "location": {"x":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },


    # x = 1
    # u(1,y)=sinh(pi)*sin(pi*y)
    {
        "location": {"x":1},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            },
            {
                "var":"y",
                "operator": lambda y: -ko.sinh(np.pi)*ko.sin(np.pi*y)
            }
        ]
    },


    # y = 0
    # u(x,0)=0
    {
        "location":{"y":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },


    # y = 1
    # u(x,1)=0
    {
        "location":{"y":1},
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
    variables=["x","y"],

    equation_structure=equation_structure,
    conditions=conditions,

    bounds={
        "x":(0,1),
        "y":(0,1)
    },

    n_samples=5000,

    model_structure=model_structure,

    epochs=50,
    batch_size=256,

    learning_rate=1e-3,

    validation_split=0.1,

    early_stopping=True,

    verbose=1,

    random_state=42
)


# ----------------------------------
# Train
# ----------------------------------

pinn.fit()


# ----------------------------------
# Evaluate
# ----------------------------------

x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

X,Y = np.meshgrid(x,y)

points = np.column_stack([
    X.ravel(),
    Y.ravel()
])

print("X shape:", X.shape)
prediction = pinn.predict(points).reshape(X.shape)


truth = (
    np.sinh(np.pi*X)
    *
    np.sin(np.pi*Y)
)


print(
    "solution MSE:",
    np.mean((prediction-truth)**2)
)


# ----------------------------------
# PDE residual
# ----------------------------------

residual = pinn._calc_pde(
    tf.constant(points,dtype=tf.float32)
)


print(
    "PDE residual:",
    tf.reduce_mean(tf.abs(residual)).numpy()
)


# ----------------------------------
# sklearn compatibility
# ----------------------------------

print("Estimator params:")
print(pinn.get_params())


print("Score:")
print(pinn.score(points))

import matplotlib.pyplot as plt


# ----------------------------------
# Plot solution comparison
# ----------------------------------

error = np.abs(prediction - truth)
print("pred shape:", prediction.shape)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,5),
    constrained_layout=True
)


# Prediction
im0 = axes[0].imshow(
    prediction,
    origin="lower",
    extent=[0,1,0,1],
    aspect="auto"
)

axes[0].set_title("DeepPINN prediction")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])


# Truth
im1 = axes[1].imshow(
    truth,
    origin="lower",
    extent=[0,1,0,1],
    aspect="auto"
)

axes[1].set_title("Analytical solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(im1, ax=axes[1])


# Error
im2 = axes[2].imshow(
    error,
    origin="lower",
    extent=[0,1,0,1],
    aspect="auto"
)

axes[2].set_title("Absolute error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(im2, ax=axes[2])


plt.show()

pinn.plot(loc={"x":0},n_samples=1000)