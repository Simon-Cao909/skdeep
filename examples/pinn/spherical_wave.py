from skdeep import DeepPINN
import numpy as np
import tensorflow.keras.ops as ko

pinn = DeepPINN(variables=['r','θ','φ','t'],
         equation_structure="u_tt - (c^2)Δ[r,θ,φ]u = 0",
         coordinates='spherical',
         functions={'u':1},
         conditions=[{'loc':{'t':0},'eqn':'u_t = 0'},
                     {'loc':{'t':0},'eqn':[
                         {'var':'u'},
                         {'var':'r','op':lambda r:ko.exp(-r ** 2),'coef':-1}
                     ]},
                     {'loc':{},'eqn':'u_θ = 0'},
                     {'loc':{},'eqn':'u_φ = 0'}],
         bounds={'r':(1,2),'θ':(np.pi/4,np.pi/2),'φ':(-np.pi,np.pi),'t':(0,1)},
         constants=[{'name':'c','val':2}],
         n_samples=1000,
         epochs=100,
         batch_size=64,

         model_structure=[['N',np.asarray([0,0,0,0]),np.asarray([1,np.pi,2*np.pi,1])],
                          ['D',64,'tanh'],
                          ['D',64,'tanh'],
                          ['D',32,'tanh'],
                          ['D',1,'linear']],
         build_setting='quick',
         early_stopping=False,
         learning_rate=1e-4)

pinn.fit()

pinn.plot(loc={'φ':np.pi,'t':0.5},function='u')