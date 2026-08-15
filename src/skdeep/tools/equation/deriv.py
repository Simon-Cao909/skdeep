import tensorflow.keras.ops as ko

def is_nabla(d):
    return (
        (d[0] == '∇'
        or any(label in d.lower() for label in ['div','divergence','curl','cross','grad','gradient']))
        and not is_laplace(d)
    )

def is_laplace(d):
    return d[0] == 'Δ' or any(label in d.lower() for label in ['laplacian','∇^2'])

def apply_grad(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i):
    if vec_val.shape[1] != 1:
        raise ValueError(f"Cannot take gradient of function with {vec_val.shape[1]} outputs. "
                            f"Must be 1.")

    if coordinates == 'cartesian':
        post_deriv = [1]*len(vs)
    elif coordinates == 'polar':
        post_deriv = [1,1/r]
    elif coordinates == 'cylindrical':
        post_deriv = [1,1/rho,1]
    elif coordinates == 'spherical':
        post_deriv = [1,1/r,1/(r*ko.sin(th))]

    for j,v in enumerate(vs):
        val = var_to_val[v]
        grad = post_deriv[j]*tape.gradient(vec_val,val)

        if grad is None:
            raise RuntimeError(f"Could not compute {i} derivative when computing gradient with respect to {v}")

        if j == 0:
            res = grad
        else:
            res = ko.concatenate([res,grad],axis=1)

    return res

def apply_div(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i):
    res = 0

    if vec_val.shape[1] != len(vs):
        raise ValueError(f"Cannot take divergence of function over {len(vs)} variables when "
                            f"the length of vector output is {vec_val.shape[1]}.")

    if coordinates == 'cartesian':
        pre_deriv = [1]*len(vs)
        post_deriv = [1]*len(vs)
    elif coordinates == 'polar':
        pre_deriv = [r,1]
        post_deriv = [1/r]*2
    elif coordinates == 'cylindrical':
        pre_deriv = [rho,1,1]
        post_deriv = [1/rho]*2+[1]
    elif coordinates == 'spherical':
        pre_deriv = [r**2,ko.sin(th),1]
        post_deriv = [1/r**2,1/(r*ko.sin(th)),1/(r*ko.sin(th))]

    for j,v in enumerate(vs):
        f_val = vec_val[:,j:j+1]
        val = var_to_val[v]

        grad = tape.gradient(f_val*pre_deriv[j],val)

        if grad is None:
            raise RuntimeError(f"Could not compute derivative with respect to {v} when computing divergence")
        
        res += grad*post_deriv[j]

    return res

def apply_curl(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i):
    if len(vs) != 3:
        raise ValueError(f"Cannot take curl of function over {len(vs)} variables. "
                            f"Must be 3.")
    
    if vec_val.shape[1] != 3:
        raise ValueError(f"Cannot take curl of function with {vec_val.shape[1]} outputs. "
                            f"Must be 3.")

    if coordinates == 'cartesian':
        scale_facs = [1]*len(vs)
    elif coordinates == 'cylindrical':
        scale_facs = [1,rho,1]
    elif coordinates == 'spherical':
        scale_facs = [1,r,r*ko.sin(th)]

    v1,v2,v3 = vs
    f1,f2,f3 = vec_val[:,0:1],vec_val[:,1:2],vec_val[:,2:3]

    f3_v2 = tape.gradient(f3*scale_facs[2],var_to_val[v2])
    f2_v3 = tape.gradient(f2*scale_facs[1],var_to_val[v3])

    f1_v3 = tape.gradient(f1*scale_facs[0],var_to_val[v3])
    f3_v1 = tape.gradient(f3*scale_facs[2],var_to_val[v1])

    f2_v1 = tape.gradient(f2*scale_facs[1],var_to_val[v1])
    f1_v2 = tape.gradient(f1*scale_facs[0],var_to_val[v2])

    labels = ["f3_v2","f2_v3","f1_v3","f3_v1","f2_v1","f1_v2"]
    for k,grad in enumerate([f3_v2,f2_v3,f1_v3,f3_v1,f2_v1,f1_v2]):
        if grad is None:
            raise RuntimeError(f"Could not compute {i} derivative to make {labels[k]}")

    curl = ko.concatenate([
        (f3_v2 - f2_v3)/(scale_facs[2]*scale_facs[1]),
        (f1_v3 - f3_v1)/(scale_facs[0]*scale_facs[2]),
        (f2_v1 - f1_v2)/(scale_facs[1]*scale_facs[0])
    ], axis=1)

    return curl

def apply_laplace(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i):
    if vec_val.shape[1] > 1:
        div = apply_div(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i)
        grad_of_div = apply_grad(d,div,coordinates,vs,r,rho,th,var_to_val,tape,i)

        curl = apply_curl(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i)
        curl_of_curl = apply_curl(d,curl,coordinates,vs,r,rho,th,var_to_val,tape,i)
        res = (grad_of_div - curl_of_curl)
    else:
        grad = apply_grad(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i)
        div_of_grad = apply_div(d,grad,coordinates,vs,r,rho,th,var_to_val,tape,i)
        res = div_of_grad

    return res

def apply_single(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i):
    val = var_to_val[d]

    grads = [
        tape.gradient(vec_val[:,j:j+1],val)
        for j in range(vec_val.shape[1])
    ]

    if any(g is None for g in grads):
        raise RuntimeError(f"Could not compute {i} derivative with respect to {d}")
    
    return ko.concatenate(grads,axis=1)

def find_deriv(i,d,var_to_val,tape,variables,derivs,ind,coordinates):
    i += 1

    if i % 10 == 1 and i % 100 != 11:
        i = f"{i}st"
    elif i % 10 == 2 and i % 100 != 12:
        i = f"{i}nd"
    elif i % 10 == 3 and i % 100 != 13:
        i = f"{i}rd"
    else:
        i = f"{i}th"

    vec_val = derivs[ind]

    if d in variables:                    
        apply = apply_single
    elif is_nabla(d) or is_laplace(d):
        # When it uses nabla, needs to be of form ∇[var1,var2,...] or Δ[var1,var2,...] for laplacian
        lbrack = d.find('[')
        rbrack = d.find(']')

        if lbrack == -1 or rbrack == -1:
            lbrack = d.find('(')
            rbrack = d.find(')')
            if lbrack == -1 or rbrack == -1:
                raise ValueError("Brackets specifying the variables need to be given when applying ∇")
        
        vs = d[lbrack+1:rbrack].split(',')
        vs = sorted(vs,key=variables.index)

        expec = {'cartesian':len(vs),'polar':2,'cylindrical':3,'spherical':3}
        if len(vs) != expec[coordinates]:
            raise ValueError("Too many variables given between the brackets when "
                             "applying multivariate derivative!\n"
                             f"Expected: {expec[coordinates]}, got: {len(vs)}.")

        rest = d[rbrack+1:]

        for v in vs:
            if v not in variables:
                raise ValueError(f"{v} is not a variable when applying deriv {d}!")

        r,th,phi,rho = None,None,None,None
        if coordinates == 'polar':
            r,_ = vs
            r = var_to_val[r]
        elif coordinates == 'cylindrical':
            rho,_,_ = vs
            rho = var_to_val[rho]
        elif coordinates == 'spherical':
            r,th,phi = vs
            r = var_to_val[r]
            th = var_to_val[th]
            phi = var_to_val[phi]

        if is_laplace(d):
            apply = apply_laplace
        elif is_nabla(d):
            if '⋅' in rest or 'dot' in rest or '●' in rest or \
                any(label in d.lower() for label in ['div','divergence']):
                apply = apply_div
            elif '×' in rest or 'cross' in rest or 'x' in rest or \
                any(label in d.lower() for label in ['curl','cross']):
                apply = apply_curl
            elif rest.strip() == '' or any(label in d.lower() for label in ['grad','gradient']):
                apply = apply_grad
    else:
        raise ValueError(f"Derivative {d} is not a variable in the given variables and is not ∇")

    derivs[ind] = apply(d,vec_val,coordinates,vs,r,rho,th,var_to_val,tape,i)