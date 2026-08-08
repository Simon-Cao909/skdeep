import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skdeep.pinn import DeepPINN


# ----------------------------------
# Model architecture
# ----------------------------------

model_structure = [
    {
        "type": "Norm",
        "mins": np.array([0]),
        "maxs": np.array([2*np.pi]),
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
        "units": 2,
        "activation": "linear"
    }
]


# ----------------------------------
# Coupled system:
#
# u' - v = 0
# v' + u = 0
#
# with
#
# u(0) = 0
# v(0) = 1
#
# Analytical solution:
#
# u(x) = sin(x)
# v(x) = cos(x)
# ----------------------------------

equation_structure = [
    [
        {
            "var": "u",
            "deriv": ["x"],
            "coef": 1
        },
        {
            "var": "v",
            "coef": -1
        }
    ],

    [
        {
            "var": "v",
            "deriv": ["x"],
            "coef": 1
        },
        {
            "var": "u",
            "coef": 1
        }
    ]
]


# ----------------------------------
# Boundary / initial conditions
# ----------------------------------

conditions = [

    # u(0) = 0
    {
        "location": {"x": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "u",
                "coef": 1
            }
        ]
    },

    # v(0) = 1
    {
        "location": {"x": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "v",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    }
]


# ----------------------------------
# Create estimator
# ----------------------------------

pinn = DeepPINN(
    variables=["x"],

    functions=["u", "v"],

    equation_structure=equation_structure,

    conditions=conditions,

    bounds={
        "x": (0, 2*np.pi)
    },

    n_samples=5000,

    model_structure=model_structure,

    epochs=100,

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

x = np.linspace(0, 2*np.pi, 500)

points = x.reshape(-1, 1)

prediction = pinn.predict(points)

u_prediction = prediction[:, 0]
v_prediction = prediction[:, 1]


# Analytical solutions

u_truth = np.sin(x)
v_truth = np.cos(x)


# ----------------------------------
# Solution errors
# ----------------------------------

u_error = np.abs(u_prediction - u_truth)
v_error = np.abs(v_prediction - v_truth)

print(
    "u MSE:",
    np.mean((u_prediction - u_truth)**2)
)

print(
    "v MSE:",
    np.mean((v_prediction - v_truth)**2)
)


# ----------------------------------
# PDE residuals
# ----------------------------------

residuals = pinn._calc_pde(
    tf.constant(points, dtype=tf.float32)
)

print(
    "u PDE residual:",
    tf.reduce_mean(tf.abs(residuals[0])).numpy()
)

print(
    "v PDE residual:",
    tf.reduce_mean(tf.abs(residuals[1])).numpy()
)


# ----------------------------------
# sklearn compatibility
# ----------------------------------

print("Estimator params:")
print(pinn.get_params())

print("Score:")
print(pinn.score(points))


# ----------------------------------
# Plot solutions
# ----------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 8),
    constrained_layout=True
)


# u prediction

axes[0, 0].plot(
    x,
    u_prediction,
    label="Prediction"
)

axes[0, 0].plot(
    x,
    u_truth,
    "--",
    label="Analytical"
)

axes[0, 0].set_title("u(x)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("u")
axes[0, 0].legend()


# u error

axes[0, 1].plot(
    x,
    u_error
)

axes[0, 1].set_title("|u prediction - u truth|")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("Absolute error")


# u residual

axes[0, 2].plot(
    x,
    np.abs(residuals[0].numpy())
)

axes[0, 2].set_title("|u PDE residual|")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("Residual")


# v prediction

axes[1, 0].plot(
    x,
    v_prediction,
    label="Prediction"
)

axes[1, 0].plot(
    x,
    v_truth,
    "--",
    label="Analytical"
)

axes[1, 0].set_title("v(x)")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("v")
axes[1, 0].legend()


# v error

axes[1, 1].plot(
    x,
    v_error
)

axes[1, 1].set_title("|v prediction - v truth|")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Absolute error")


# v residual

axes[1, 2].plot(
    x,
    np.abs(residuals[1].numpy())
)

axes[1, 2].set_title("|v PDE residual|")
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("Residual")


plt.show()