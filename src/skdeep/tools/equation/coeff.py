import numpy as np
import tensorflow.keras.ops as ko

def parse_scalar_coef(coef,var_to_val,func_to_val,constants):
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
        skip = False
        for c_ind,char in enumerate(coef):
            if skip:
                skip = False
                continue

            to_apply = None
            op_change = False

            if char in func_to_val:
                if c_ind + 1 < len(coef) and isinstance(coef[c_ind+1],int):
                    to_apply = func_to_val[char+str(coef[c_ind+1])]
                    skip = True
                else:
                    to_apply = func_to_val[char]

                if to_apply.shape[1] != 1:
                    raise ValueError(f"Coefficient {char} cannot be a vector-valued function.")
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
                op_change = True
            elif char == '/':
                coef_operator = 'div'
                op_change = True
            elif char == ' ':
                coef_operator = 'mult'
                op_change = True

            if to_apply is not None:
                if coef_operator == 'pow':
                    cfs **= to_apply
                elif coef_operator == 'div':
                    cfs /= to_apply
                elif coef_operator == 'mult':
                    cfs *= to_apply
            elif not op_change:
                raise ValueError(f"Unknown character in coefficient: {char}")

        cfs *= sign

    return cfs

def parse_vector_coef(coef,var_to_val,func_to_val,constants):
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
        extra_l = parse_scalar_coef(extra_l,var_to_val,func_to_val,constants)

    if len(extra_r) != 0:
        extra_r = parse_scalar_coef(extra_r,var_to_val,func_to_val,constants)

    cfs = ko.concatenate([
        parse_scalar_coef(el)
        for el in comps
    ],axis=1)

    return extra_l*cfs*extra_r