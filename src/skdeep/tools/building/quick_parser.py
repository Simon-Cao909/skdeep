import re

def safe_get(lt,ind,fallback=None):
    '''
    Safely gets an element from a list.

    If the index is out of the max index,
    the fallback will be returned.

    Parameters
    ----------
    lt : list
        The list to get the element from.
    
    ind : int
        The index of the element.
    
    fallback : anything
        The fallback if the index is not valid.
    
    Returns
    -------
    Anything
        The element of the list at that index 
        or the fallback if the index is out of range.
    '''
    if -len(lt) < ind < len(lt):
        return lt[ind]
    else:
        return fallback

def parse_quick(structs):
    '''
    Parses model_structure when build_setting = 'quick'

    Parameters
    ----------
    structs : list
        A list of lists describing the model structure
    
    Returns
    -------
    dict
        A model_structure dictionary after parsing.

        It is of the form when build_setting = 'normal'
    '''
    new_structure = []

    for struct in structs:
        layer_type = safe_get(struct,0)

        ### SIMPLE LAYERS ###
        if layer_type == "D" or layer_type.lower() == 'dense':
            units = safe_get(struct,1)
            activation = safe_get(struct,2)
            new_structure.append({'type':layer_type,'units':units,'activation':activation})
        elif layer_type == 'd' or layer_type.lower() == 'dropout':
            rate = safe_get(struct,1)
            new_structure.append({'type':layer_type,'rate':rate})
        elif layer_type in ['C','CT'] or layer_type.lower() in ['conv','convolution','conv2d']+\
                                                               ['conv_transpose','convolution_transpose','conv2dtranspose']:
            filters = safe_get(struct,1)
            kernel_size = safe_get(struct,2)
            activation = safe_get(struct,3)

            default_stride = tuple([1]*len(kernel_size))
            
            strides = safe_get(struct,4,default_stride)
            padding = safe_get(struct,5,"valid")
            data_format = safe_get(struct,6)
            new_structure.append({'type':layer_type,'filters':filters,'kernel_size':kernel_size,
                                  'strides':strides,'padding':padding,'data_format':data_format})
        elif layer_type == 'GN' or layer_type.lower() in ['group_norm','group_normalization']:
            groups = safe_get(struct,1,32)
            axis = safe_get(struct,2,-1)
            epsilon = safe_get(struct,3,0.001)
            center = safe_get(struct,4,True)
            scale = safe_get(struct,5,True)
            new_structure.append({'type':layer_type,'groups':groups,'axis':axis,'epsilon':epsilon,
                                  'center':center,'scale':scale})
        elif layer_type == 'BN' or layer_type.lower() in ['batch_norm','batch_normalization']:
            axis = safe_get(struct,1,-1)
            momentum = safe_get(struct,2,0.99)
            epsilon = safe_get(struct,3,0.001)
            center = safe_get(struct,4,True)
            scale = safe_get(struct,5,True)
            new_structure.append({'type':layer_type,'axis':axis,'momentum':momentum,'epsilon':epsilon,
                                 'center':center,'scale':scale})
        elif layer_type == 'MP' or layer_type.lower() == 'max_pooling':
            pool_size = safe_get(struct,1,(2,2))
            strides = safe_get(struct,2)
            padding = safe_get(struct,3,"valid")
            data_format = safe_get(struct,4)
            new_structure.append({'type':layer_type,'pool_size':pool_size,'strides':strides,
                                  'padding':padding,'data_format':data_format})
        elif layer_type in ['GAP','F'] or layer_type.lower() in ['global_avg_pooling','global_average_pooling']+\
                                                                ['flat','flatten']:
            data_format = safe_get(struct,1)
            new_structure.append({'type':layer_type,'data_format':data_format})
        elif layer_type == 'UP' or layer_type.lower() in ['upsampling','upsample','upsampling2d']:
            size = safe_get(struct,1,(2,2))
            data_format = safe_get(struct,2)
            new_structure.append({'type':layer_type,'size':size,'data_format':data_format})
        elif layer_type == 'N' or layer_type.lower() in ['normalization','norm']:
            mins = safe_get(struct,1)
            maxs = safe_get(struct,2)
            new_structure.append({'type':layer_type,'mins':mins,'maxs':maxs})
        elif layer_type.lower() == 'custom':
            layer = safe_get(struct,1)
            new_structure.append({'type':layer_type,'layer':layer})
        
        ### SPECIAL BLOCKS ###
        elif layer_type == 'R' or layer_type.lower() in ['resnet','residual']:
            layers = parse_quick(safe_get(struct,1))
            final_activation = safe_get(struct,2,'linear')
            allow_projection = safe_get(struct,3,True)
            new_structure.append({'type':layer_type,'layers':layers,
                                  'final_activation':final_activation,
                                  'allow_projection':allow_projection})
        elif layer_type == 'I' or layer_type.lower() in ['inception','incep','multi-output']:
            branches = [parse_quick(branch) for branch in safe_get(struct,1)]
            new_structure.append({'type':layer_type,'branches':branches})
        elif layer_type == 'X' or layer_type.lower() in ['xcep','xception']:
            xcep_specs = safe_get(struct,1)

            new_xcep_specs = []
            for spec in xcep_specs:
                filters = safe_get(spec,0)
                kernel_size = safe_get(spec,1)
                activation = safe_get(spec,2)
                padding = safe_get(spec,3,"same")
                new_xcep_specs.append({'filters':filters,'kernel_size':kernel_size,
                                       'activation':activation,'padding':padding})

            final_activation = safe_get(struct,2,'linear')
            allow_projection = safe_get(struct,3,True)
            new_structure.append({'type':layer_type,'xcep_specs':new_xcep_specs,
                                  'final_activation':final_activation,
                                  'allow_projection':allow_projection})
        elif layer_type.lower() == 'regressor':
            model = safe_get(struct,1)
            new_structure.append({'type':layer_type,'model':model})
        elif layer_type == 'NN' or layer_type.lower() == 'neural':
            model = safe_get(struct,1)
            freeze = safe_get(struct,2,False)
            new_structure.append({'type':layer_type,'model':model,'freeze':freeze})
        elif layer_type.lower() == 'multi-input':
            branches = [parse_quick(branch) for branch in safe_get(struct,1)]
            merge_layer = safe_get(struct,2)
            new_structure.append({'type':layer_type,'branches':branches,'merge_layer':merge_layer})
        else:
            new_structure.append({'type':layer_type})
    
    return new_structure

def split_terms(eqn):
    parts = []
    depth = 0
    start = 0

    for i, c in enumerate(eqn):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c in "+-" and depth == 0 and i > start:
            parts.append(eqn[start:i])
            start = i

    parts.append(eqn[start:])
    return parts

def find_one(string,lt):
    for el in lt:
        find = string.find(el)
        if find != -1:
            return find

    return -1
        

def parse_eqn(eqn):
    eqn = eqn.replace(" ","")
    sides = eqn.split("=")

    if len(sides) == 1:
        parts = split_terms(eqn)
    elif len(sides) == 2:
        lhs,rhs = sides
        l_parts = split_terms(lhs)
        r_parts = split_terms(rhs)

        if rhs == '0':
            parts = l_parts
        else:
            for p in r_parts:
                if p[0] == '-':
                    l_parts.append(p.lstrip('-'))
                elif p[0] == '+':
                    l_parts.append('-'+p.lstrip('+'))
                else:
                    l_parts.append('-'+p)

            parts = l_parts
    else:
        raise ValueError("More than one '=' used in equation")
    
    parts = [p.strip("+") for p in parts]

    structure = []

    for term in parts:
        sign = '-' if term.startswith('-') else ''
        term = term.lstrip("+-")
        
        l_brack = term.find("(")
        r_brack = term.find(")")

        if (l_brack == -1 and r_brack == -1) or (r_brack == 1 + l_brack):
            cf = -1 if sign == '-' else 1
        elif l_brack == -1 or r_brack == -1:
            raise ValueError(f"left bracket found: {True if l_brack != -1 else False}\n"
                             f"right bracket found: {True if r_brack != -1 else False}")
        else:
            cf = sign+term[l_brack+1:r_brack]

        focus = term[r_brack+1:]

        if len(focus) == 0:
            var = 'const'

        l_brack2 = focus.find("(")
        r_brack2 = focus.find(")")

        if l_brack2 == -1 and r_brack2 == -1:
            operator = 'identity'
        elif l_brack2 == -1 or r_brack2 == -1:
            raise ValueError(f"left bracket found: {True if l_brack2 != -1 else False}\n"
                             f"right bracket found: {True if r_brack2 != -1 else False}")
        else:
            operator = focus[:l_brack2]
            focus = focus[l_brack2+1:r_brack2]

        focus = focus.split("_")
        var = focus[0]
        derivs = []

        nabla_i = find_one(var,['∇^2','∇','Δ'])
        if nabla_i != -1:
            end_sym = find_one(var,['⋅','●','×',']',')'])
            derivs.append(var[nabla_i:end_sym+1])
            var = var[end_sym+1:]

        if len(focus) == 1:
            focus.append("")

        derivs += list(focus[1])
        structure.append({
            'var':var,
            'deriv':derivs,
            'coef':cf,
            'op':operator
        })

    return structure