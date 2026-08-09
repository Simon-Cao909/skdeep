import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skdeep.pinn import DeepPINN


# ============================================================
# SIMPLE DIVERGENCE-FREE VECTOR PINN
# ============================================================
#
# Solve
#
#               ∇ · E = 0
#
# on the unit cube:
#
#               0 <= x,y,z <= 1
#
# with boundary condition
#
#               E = (1, 0, 0)
#
# everywhere on the boundary.
#
# Exact solution:
#
#               E(x,y,z) = (1, 0, 0)
#
# and therefore
#
#               ∇ · E = 0.
#
# ============================================================


# ============================================================
# MODEL
# ============================================================

model_structure = [

    {
        "type": "Norm",
        "mins": np.array([0, 0, 0]),
        "maxs": np.array([1, 1, 1]),
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
        "units": 32,
        "activation": "tanh"
    },

    # Vector field:
    #
    # E = (Ex, Ey, Ez)

    {
        "type": "Dense",
        "units": 3,
        "activation": "linear"
    }
]


# ============================================================
# PDE
# ============================================================
#
# ∇ · E = 0
#
# IMPORTANT:
#
# The variables inside ∇ are x,y,z.
# The function being differentiated is u.
#
# ============================================================

equation_structure = [
    {
        "var": "u",
        "derivatives": ["∇(x,y,z)⋅"]
    }
]


# ============================================================
# BOUNDARY CONDITIONS
# ============================================================
#
# We want:
#
#       Ex = 1
#       Ey = 0
#       Ez = 0
#
# on every boundary face.
#
# Since the framework represents the vector-valued function
# as one function "u", we can impose each component separately
# using u1, u2, u3.
#
# ============================================================

conditions = [

    # --------------------------------------------------------
    # x = 0
    # --------------------------------------------------------

    {
        "location": {"x": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u1",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    },

    {
        "location": {"x": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u2",
                "coef": 1
            }
        ]
    },

    {
        "location": {"x": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u3",
                "coef": 1
            }
        ]
    },


    # --------------------------------------------------------
    # x = 1
    # --------------------------------------------------------

    {
        "location": {"x": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u1",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    },

    {
        "location": {"x": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u2",
                "coef": 1
            }
        ]
    },

    {
        "location": {"x": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u3",
                "coef": 1
            }
        ]
    },


    # --------------------------------------------------------
    # y = 0
    # --------------------------------------------------------

    {
        "location": {"y": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u1",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    },

    {
        "location": {"y": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u2",
                "coef": 1
            }
        ]
    },

    {
        "location": {"y": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u3",
                "coef": 1
            }
        ]
    },


    # --------------------------------------------------------
    # y = 1
    # --------------------------------------------------------

    {
        "location": {"y": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u1",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    },

    {
        "location": {"y": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u2",
                "coef": 1
            }
        ]
    },

    {
        "location": {"y": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u3",
                "coef": 1
            }
        ]
    },


    # --------------------------------------------------------
    # z = 0
    # --------------------------------------------------------

    {
        "location": {"z": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u1",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    },

    {
        "location": {"z": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u2",
                "coef": 1
            }
        ]
    },

    {
        "location": {"z": 0},
        "n_samples": 200,

        "equation": [
            {
                "var": "u3",
                "coef": 1
            }
        ]
    },


    # --------------------------------------------------------
    # z = 1
    # --------------------------------------------------------

    {
        "location": {"z": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u1",
                "coef": 1
            },
            {
                "var": "const",
                "coef": -1
            }
        ]
    },

    {
        "location": {"z": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u2",
                "coef": 1
            }
        ]
    },

    {
        "location": {"z": 1},
        "n_samples": 200,

        "equation": [
            {
                "var": "u3",
                "coef": 1
            }
        ]
    }
]


# ============================================================
# CREATE PINN
# ============================================================

pinn = DeepPINN(

    variables=["x", "y", "z"],

    functions=["u"],

    equation_structure=equation_structure,

    conditions=conditions,

    bounds={
        "x": (0, 1),
        "y": (0, 1),
        "z": (0, 1)
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


# ============================================================
# TRAIN
# ============================================================

pinn.fit()


# ============================================================
# EVALUATE
# ============================================================

n = 20

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
z = np.linspace(0, 1, n)

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing="ij"
)

points = np.column_stack([
    X.ravel(),
    Y.ravel(),
    Z.ravel()
])


prediction = pinn.predict(points)

print()
print("=" * 70)
print("PREDICTION")
print("=" * 70)

print("Prediction shape:", prediction.shape)

print("Expected shape:", (len(points), 3))


# ============================================================
# EXACT SOLUTION
# ============================================================

truth = np.zeros_like(prediction)

truth[:, 0] = 1.0


# ============================================================
# SOLUTION ERROR
# ============================================================

error = prediction - truth

print()
print("=" * 70)
print("SOLUTION ERROR")
print("=" * 70)

print(
    "Ex MSE:",
    np.mean(error[:, 0] ** 2)
)

print(
    "Ey MSE:",
    np.mean(error[:, 1] ** 2)
)

print(
    "Ez MSE:",
    np.mean(error[:, 2] ** 2)
)

print(
    "Total MSE:",
    np.mean(error ** 2)
)


# ============================================================
# DIVERGENCE RESIDUAL
# ============================================================

residual = pinn._calc_pde(
    tf.constant(points, dtype=tf.float32)
)

divergence = residual[0]


print()
print("=" * 70)
print("DIVERGENCE")
print("=" * 70)

print(
    "Mean absolute divergence:",
    tf.reduce_mean(tf.abs(divergence)).numpy()
)

print(
    "Maximum absolute divergence:",
    tf.reduce_max(tf.abs(divergence)).numpy()
)


# ============================================================
# CHECK THE NETWORK AT A FEW RANDOM POINTS
# ============================================================

test_points = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.7, 0.2],
    [0.8, 0.3, 0.9],
    [0.5, 0.5, 0.5],
])

test_prediction = pinn.predict(test_points)

print()
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for point, pred in zip(test_points, test_prediction):

    print(
        f"(x,y,z) = {point}"
        f"  ->  E = {pred}"
    )


# ============================================================
# SCORE
# ============================================================

print()
print("=" * 70)
print("PINN SCORE")
print("=" * 70)

print(pinn.score(points))


# ============================================================
# PLOT ONE SLICE
# ============================================================

z_index = np.argmin(np.abs(z - 0.5))

Ex = prediction[:, 0].reshape(X.shape)
Ey = prediction[:, 1].reshape(X.shape)
Ez = prediction[:, 2].reshape(X.shape)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5),
    constrained_layout=True
)

for ax, field, title in zip(
    axes,
    [
        Ex[:, :, z_index],
        Ey[:, :, z_index],
        Ez[:, :, z_index]
    ],
    [
        "$E_x$ at z=0.5",
        "$E_y$ at z=0.5",
        "$E_z$ at z=0.5"
    ]
):

    im = ax.imshow(
        field,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.colorbar(im, ax=ax)

plt.show()
