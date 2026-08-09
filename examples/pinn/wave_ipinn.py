import numpy as np
import tensorflow as tf
from tensorflow.keras import ops as ko

from skdeep.pinn import DeepPINN

variables = ['x','t']

bounds = {
    'x': (0,1),
    't': (0,1)
}

constants = [{
    "name": "c",
    "value": 0.5,        # intentionally wrong
    "trainable": True
}]

c_true = 2.0

N = 500

x = np.random.rand(N)
t = np.random.rand(N)

u = np.sin(np.pi*x) * np.cos(np.pi*c_true*t)

data = np.column_stack([x, t, u])
data = tf.cast(data,tf.float32)

equation_structure = "u_tt - (c^2)u_xx"

conditions = [
    {'loc':{'t':0},
     'eqn':[
         {},
         {'var':'x',
          'operator':lambda x: -ko.sin(np.pi*x)}
     ]},
    {'loc':{'t':0},
     'eqn':[
         {'deriv':['t']}
     ]},
    {'loc':{'x':0},
     'eqn':[{}]},
    {'loc':{'x':1},
     'eqn':[{}]},
]

model_structure = [
    ['N',np.array([0,0]),np.array([1,1])],
    ['D',64,'tanh'],
    ['D',64,'tanh'],
    ['D',32,'tanh'],
    ['D',1,'linear'],
]

model = DeepPINN(variables=variables,
                    equation_structure=equation_structure,
                    conditions=conditions,
                    bounds=bounds,
                    n_samples=500,
                    constants=constants,
                    data=data,

                    model_structure=model_structure,
                    build_setting='quick',
                    epochs=100,
                    learning_rate=1e-3,
                    batch_size=32,
                    validation_split=0.1,
                    early_stopping=True,
                    verbose=1,
                    random_state=42,
                    loss_weighting={'pde':1,'conditions':10,'data':100})

model.fit()

print(model.constants_) # 1.98 Pretty good!