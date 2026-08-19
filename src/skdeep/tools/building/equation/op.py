import tensorflow.keras.ops as ko
from collections.abc import Callable

str_to_op = {
    'identity':lambda x: x,
    'sin':ko.sin,
    'sinh':ko.sinh,
    'cos':ko.cos,
    'cosh':ko.cosh,
    'tan':ko.tan,
    'tanh':ko.tanh,
    'ln':ko.log,
    'square':ko.square,
    'sqrt':ko.sqrt,
}

def get_op(operator):
    if isinstance(operator,Callable):
        return operator
    elif isinstance(operator,str):
        if operator not in str_to_op:
            raise ValueError(
                f"If string, operator must be in {list(str_to_op.keys())}"
            )

        name = operator
        operator = lambda var: str_to_op[name](var)
    else:
        raise ValueError("Operator given of unknown type.\n"
                         f"Operator: {operator}, type: {type(operator)}\n"
                         "Expected type callable or string")

    return operator