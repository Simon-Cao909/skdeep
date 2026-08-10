from skdeep import DeepPINN
import numpy as np
import tensorflow.keras.ops as ko

variables = ["x", "y", "z", "t"]
functions = ["E", "B"]

constants = [
    {
        "name": "ε",
        "value": 1,
        "trainable": False,
    },
    {
        "name": "μ",
        "value": 1,
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
        "location": {"t": 0},
        "n_samples": 200,
        "equation": "E1 - ()sin(y) = 0"
    },

    # E2(x,y,z,0) = 0
    {
        "location": {"t": 0},
        "n_samples": 200,
        "equation": "E2 = 0"
    },

    # E3(x,y,z,0) = 0
    {
        "location": {"t": 0},
        "n_samples": 200,
        "equation": "E3 = 0"
    },

    # B1(x,y,z,0) = 0
    {
        "location": {"t": 0},
        "n_samples": 200,
        "equation": "B1 = 0"
    },

    # B2(x,y,z,0) = 0
    {
        "location": {"t": 0},
        "n_samples": 200,
        "equation": "B2 = 0"
    },

    # B3(x,y,z,0) = -sin(y)
    {
        "location": {"t": 0},
        "n_samples": 200,
        "equation": "B3 = -()sin(y)"
    }
]

bounds = {
    "x": (0, 2 * np.pi),
    "y": (0, 2 * np.pi),
    "z": (0, 2 * np.pi),
    "t": (0, 1)
}

model_structure = [
    [
        "N",
        np.asarray([0, 0, 0, 0]),
        np.asarray([2*np.pi, 2*np.pi, 2*np.pi, 1])
    ],
    ["D", 96, "tanh"],
    ["D", 96, "tanh"],
    ["D", 64, "tanh"],
    [
        "multi-output",
        [
            [["D", 3, "linear"]],  # E
            [["D", 3, "linear"]]   # B
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


X = np.asarray([[1,2,0.5,0.3]])
print(X.shape)
pred = pinn.predict(X)

print("E: ", pred[0])
print("B: ", pred[1])
print("Score:", pinn.score())