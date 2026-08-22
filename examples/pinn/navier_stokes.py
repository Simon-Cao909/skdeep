from skdeep import DeepPINN

pinn = DeepPINN(variables=['x','y','z','t'],
                equation_structure=[
                    "ρ_t + ∇⋅(ρV) = 0",
                    "(ρV1)_t + ∇⋅(ρV*V1) = -ρ_x + (1/R)∇⋅a",
                    "(ρV2)_t + ∇⋅(ρV*V2) = -ρ_y + (1/R)∇⋅b",
                    "(ρV3)_t + ∇⋅(ρV*V3) = -ρ_z + (1/R)∇⋅c",

                    "E_t + ∇⋅(EV) + ∇⋅(pV) + (1/RP)∇⋅q = \
                    (1/R)(a⋅V)_x + \
                    (1/R)(b⋅V)_y + \
                    (1/R)(c⋅V)_z"
                ],
                functions={'ρ':1,'V':3,'a':3,'b':3,'c':3,'E':1,'q':3,'p':1},
                constants=[{'name':'R','val':1},{'name':'P','val':1}],
                bounds={'x':(0,1),'y':(0,1),'z':(0,1),'t':(0,1)},
                conditions=[
                    {'loc':{'t':0},
                    'eqn':[{'var':'V1'},{'var':'x','op':'sin','coef':[{'var':'y','op':'cos','coef':[{'var':'z','op':'cos','coef':'-1'}]}]}]},
                    {'loc':{'t':0},
                     'eqn':[{'var':'V2'},{'var':'x','op':'cos','coef':[{'var':'y','op':'sin','coef':[{'var':'z','op':'cos'}]}]}]},
                    {'loc':{'t':0},
                     'eqn':[{'var':'V3'}]},
                    {'loc':{'t':0},
                     'eqn':[{'var':'ρ'},{'var':'1','coef':'-1'}]}
                ],
                n_samples=1000,
                batch_size=64,
                epochs=50)

pinn.fit()
pinn.plot(loc={'t':0.5},function='V',vec_el=None,draw=True,n_samples=5)