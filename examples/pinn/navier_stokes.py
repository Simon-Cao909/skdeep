from skdeep import DeepPINN

pinn = DeepPINN(variables=['x','y','z','t'],
                equation_structure=[
                    [{'var':'ρ','deriv':['t']},
                    {'var':'V','coef':'ρ','deriv':['∇[x,y,z]⋅'],'apply_coef':'before deriv'}],

                    [{'var':'ρ','coef':'V1','deriv':['t'],'apply_coef':'before deriv'},
                     {'var':'V','coef':'ρV1','deriv':['∇[x,y,z]⋅'],'apply_coef':'before deriv'},
                     {'var':'ρ','deriv':['x']},
                     {'var':'a','coef':'-1/R','deriv':['∇[x,y,z]⋅']}],

                    [{'var':'ρ','coef':'V2','deriv':['t'],'apply_coef':'before deriv'},
                     {'var':'V','coef':'ρV2','deriv':['∇[x,y,z]⋅'],'apply_coef':'before deriv'},
                     {'var':'ρ','deriv':['y']},
                     {'var':'b','coef':'-1/R','deriv':['∇[x,y,z]⋅']}],

                    [{'var':'ρ','coef':'V3','deriv':['t'],'apply_coef':'before deriv'},
                     {'var':'V','coef':'ρV3','deriv':['∇[x,y,z]⋅'],'apply_coef':'before deriv'},
                     {'var':'ρ','deriv':['z']},
                     {'var':'c','coef':'-1/R','deriv':['∇[x,y,z]⋅']}],

                    [{'var':'E','deriv':['t']},
                     {'var':'V','coef':'E','apply_coef':'before deriv','deriv':['∇[x,y,z]⋅']},
                     {'var':'V','coef':'p','apply_coef':'before deriv','deriv':['∇[x,y,z]⋅']},
                     {'var':'q','coef':'1/RP','deriv':['∇[x,y,z]⋅']}] + \
                    [{'var':f'a{i}','coef':f'-V{i}/R','deriv':['x'],'apply_coef':'before deriv'} for i in range(1,4)] + \
                    [{'var':f'b{i}','coef':f'-V{i}/R','deriv':['y'],'apply_coef':'before deriv'} for i in range(1,4)] + \
                    [{'var':f'c{i}','coef':f'-V{i}/R','deriv':['z'],'apply_coef':'before deriv'} for i in range(1,4)]
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