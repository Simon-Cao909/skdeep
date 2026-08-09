from skdeep import DeepPINN
import numpy as np
import tensorflow.keras.ops as ko

variables = ["x", "y", "z", "t"]
functions = ['E','B','J','ρ']

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
    
    # ∇ · E - ρ/ε = 0
    [
        {
            "var": "E",
            "derivatives": ["∇(x,y,z)⋅"]
        },
        {
            "var": "ρ",
            "coef": "-1/ε"
        }
    ],

    # ∇ · B = 0
    [
        {
            "var": "B",
            "derivatives": ["∇(x,y,z)⋅"]
        }
    ],

    # ∇ × E + ∂B/∂t = 0
    [
        {
            "var": "E",
            "derivatives": ["∇(x,y,z)x"]
        },
        {
            "var": "B",
            "derivatives": ["t"]
        }
    ],

    # ∇ × B - μJ - με ∂E/∂t = 0
    [
        {
            "var": "B",
            "derivatives": ["∇(x,y,z)x"]
        },
        {
            "var": "J",
            "coef": "-μ"
        },
        {
            "var": "E",
            "derivatives": ["t"],
            "coef": "-με"
        }
    ]
]

conditions = [

    # E1(x,y,z,0) = sin(y)
    {
        "location": {"t": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "E1",
                "coef": 1
            },
            {
                "var": "y",
                "op": lambda y: -ko.sin(y)
            }
        ]
    },

    # E2(x,y,z,0) = sin(z)
    {
        "location": {"t": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "E2",
                "coef": 1
            },
            {
                "var": "z",
                "op": lambda z: -ko.sin(z)
            }
        ]
    },

    # E3(x,y,z,0) = sin(x)
    {
        "location": {"t": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "E3",
                "coef": 1
            },
            {
                "var": "x",
                "op": lambda x: -ko.sin(x)
            }
        ]
    },

    # B1(x,y,z,0) = cos(y)
    {
        "location": {"t": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "B1",
                "coef": 1
            },
            {
                "var": "y",
                "op": lambda y: -ko.cos(y)
            }
        ]
    },

    # B2(x,y,z,0) = cos(z)
    {
        "location": {"t": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "B2",
                "coef": 1
            },
            {
                "var": "z",
                "op": lambda z: -ko.cos(z)
            }
        ]
    },

    # B3(x,y,z,0) = cos(x)
    {
        "location": {"t": 0},
        "n_samples": 100,
        "equation": [
            {
                "var": "B3",
                "coef": 1
            },
            {
                "var": "x",
                "op": lambda x: -ko.cos(x)
            }
        ]
    }
]

bounds = {
    "x": (0, 2 * np.pi),
    "y": (0, 2 * np.pi),
    "z": (0, 2 * np.pi),
    "t": (0, 1)
}

model_structure = [
    ['N',np.asarray([0,0,0,0]),np.asarray([2*np.pi,2*np.pi,2*np.pi,1])],
    ['D',64,'tanh'],
    ['D',64,'tanh'],
    ['D',64,'tanh'],
    ['multi-output',[
        [['D',3,'linear']],
        [['D',3,'linear']],
        [['D',3,'linear']],
        [['D',1,'linear']]
    ]]
]

pinn = DeepPINN(variables=variables,
                equation_structure=equation_structure,
                conditions=conditions,
                bounds=bounds,
                n_samples=500,
                constants=constants,
                functions=functions,
                loss_weighting={'pde':1.0,'conditions':2.0},

                model_structure=model_structure,
                build_setting='quick',
                epochs=100,
                early_stopping=True,
                optimizer='adam')

pinn.fit()

pred = pinn.predict()
print(pred)

print(pinn.score())