import numpy as np
import tensorflow as tf
from tensorflow import is_tensor,reduce_sum
import tensorflow.keras.ops as ko
from numbers import Number

def parse_scalar_term(coef,var_to_val,func_to_val,constants):
    if coef.isnumeric():
        cfs = int(coef)
    else:
        if coef.startswith("-"):
            sign = -1
            coef = coef[1:]
        else:
            sign = 1

        cfs = 1
        coef_operator = 'mult'
        after = 'mult'
        skip = False
        for c_ind,char in enumerate(coef):
            if skip:
                skip = False
                continue

            to_apply = None
            op_change = False

            if char in func_to_val:
                if c_ind + 1 < len(coef) and coef[c_ind+1].isnumeric():
                    to_apply = func_to_val[char+str(coef[c_ind+1])]
                    skip = True
                else:
                    to_apply = func_to_val[char]
            elif char in var_to_val:
                to_apply = var_to_val[char]
            elif char in constants:
                to_apply = constants[char]
            elif char.isnumeric():
                to_apply = int(char)
            elif char == 'π':
                to_apply = np.pi
            elif char == 'e':
                to_apply = np.e

            elif char == '^':
                coef_operator = 'pow'
                after = 'pow'
                op_change = True
            elif char == '/':
                coef_operator = 'div'
                after = 'div'
                op_change = True
            elif char in [' ','*']:
                coef_operator = 'mult'
                after = 'mult'
                op_change = True
            elif char == '+':
                coef_operator = 'plus'
                after = 'mult'
                op_change = True
            elif char == '-':
                coef_operator = 'minus'
                after = 'mult'
                op_change = True
            elif char in ['⋅','●']:
                coef_operator = 'dotprod'
                after = 'mult'
                op_change = True

            if to_apply is not None:
                if coef_operator == 'pow':
                    cfs **= to_apply
                elif coef_operator == 'div':
                    cfs /= to_apply
                elif coef_operator == 'mult':
                    cfs *= to_apply
                elif coef_operator in ['plus','minus','dotprod']:
                    if is_tensor(to_apply) and not is_tensor(cfs):
                        raise ValueError("When apply plus, minus, or dotprod, and the current term is a tensor, "
                                        "the previous term must also be a tensor!")

                    if is_tensor(cfs) and cfs.shape != to_apply.shape:
                        raise ValueError("Previous term and current term shapes must match! "
                                        f"Previous term shape: {cfs.shape} | Current term shape: {to_apply.shape}")
                    
                    if coef_operator == 'plus':
                        cfs += to_apply
                    elif coef_operator == 'minus':
                        cfs -= to_apply
                    elif coef_operator == 'dotprod':
                        if not is_tensor(cfs) or not is_tensor(to_apply):
                            raise ValueError("Dot product only applies between tensors!")

                        # cfs would be of shape (n_samples,n_dim)
                        # to_apply would be of shape (n_samples,n_dim)
                        cfs = reduce_sum(cfs*to_apply, axis=1, keepdims=True)

                coef_operator = after
            elif not op_change:
                raise ValueError(f"Unknown character in coefficient: {char}")

        cfs *= sign

    return cfs

def parse_vector_term(coef,var_to_val,func_to_val,constants):
    lbrack = coef.find("(")
    rbrack = coef.find(")")

    if lbrack == -1 or rbrack == -1:
        lbrack = coef.find("[")
        rbrack = coef.find("]")
        if lbrack == -1 or rbrack == -1:
            raise ValueError("Both brackets must be given when using vector coefficients")

    comps = coef[lbrack+1:rbrack].split(",")
    if len(comps) <= 1:
        raise ValueError(f"Coefficient vector cannot be empty or one-element only. Coefficient given: {coef}")

    extra_l = coef[:lbrack]
    extra_r = coef[rbrack+1:]
    if len(extra_l) != 0:
        extra_l = parse_scalar_term(extra_l,var_to_val,func_to_val,constants)

    if len(extra_r) != 0:
        extra_r = parse_scalar_term(extra_r,var_to_val,func_to_val,constants)

    cfs = ko.concatenate([
        parse_scalar_term(el)
        for el in comps
    ],axis=1)

    return extra_l*cfs*extra_r

def parse_term(coef,var_to_val,func_to_val,constants,calc_eqn,X):
    if isinstance(coef,Number):
        cfs = coef
    elif isinstance(coef,str):
        parser = parse_vector_term if ',' in coef else parse_scalar_term
        cfs = parser(coef,var_to_val,func_to_val,constants)
    elif isinstance(coef,(list,tuple)):
        cfs = calc_eqn(X,coef)
    else:
        raise ValueError(f"Unknown coefficient type {type(coef)}")

    return cfs