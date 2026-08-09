# For equation_structure

This supports both scalar and vector equations. However, don't do things that are mathematically impossible like putting a scalar function and vector function in the same equation

## FOR SINGULAR EQUATIONS

## If list or tuple

equation_structure must be a list or tuple of dictionaries. The nth dictionary in this list or tuple denotes the nth term in the equation. When thinking of the equation, put all terms on one side so the other side is zero.

Each dictionary in this list can have keys and values:
- KEY: 'variable' or 'var' (opt | default='u')
    - The value should be a string that is equal to the variable that this term will be focusing on.
    - This variable can either be in self.variables or it can be in self.functions, where the focus will be the function. It can also be 'const', where then 1 will be the focus (or (1,1,...) for vector equations)
        - For vector-valued functions, you can use fi as the variable name, where f is the name of the function and i corresponds to the ith element in the vector. This numbering starts at 1 (as opposed to 0)
    - If this key is not included, the default variable will be self.functions[0]
    - Ex. {'variable':'x',...}, {'variable':'u1',...}
- KEY: 'derivatives' or 'deriv' (opt | default=[])
    - The value should be a list containing variables or a "special string" (see *)
    - This list will be iterated over, applying each derivative inside
        - If the current element is a variable, a derivative will be taken with respect to that variable
        - \* If the current element is a special string, it can be either of the form:
            - ∇(var1,var2,...)⋅ or 'div(var1,var2,...)' or 'divergence(var1,var2,...)'
                - The divergence will be taken. This only works if the focus is a vector valued function
            - ∇(var1,var2,...)× or ∇(var1,var2,...)x or 'curl(var1,var2,...)' or 'cross(var1,var2,...)'
                - The curl will be taken. This only works if the focus is a vector valued function
            - ∇(var1,var2,...) or 'grad(var1,var2,...)' or 'gradient(var1,var2,...)'
                - The gradient will be taken. This only works if the focus is a scalar function
            - var1, var2, ... are variables that the operator will be taken with respect to
    - This should only be included if the value associated with 'variable' was a function or is of the form fi, where f is the name of a vector-valued function and i is its ith component
        - If the list has a nonzero length when the variable was not a function, an error will be raised
    - If this key is not included, the default will be [], meaning no derivatives will be taken
    - Ex. {'derivatives':['x','x','t'],...}, {'derivatives':['t','∇(x,y,z)⋅'],...}
- KEY: 'coefficient' or 'coef' (opt | default=1)
    - The value should either be a number, string, or of the form equation_structure
        - Note that this currently does not support the equation string parsing you see below.
    - This is the coefficient of the term that will by default be applied after the operator (see below). It can either be a scalar or a vector.
        - The multiplication is always element-wise, so if the coefficient is a vector and the function is vector-valued, then it will be an element-wise multiplication as opposed to a dot product
    - If a number was given, the number will simply be the coefficient
    - If a string was given:
        - For scalar coefficients:
            - The string can start with '-', where then the coefficient will just be negative
            - If the string is just numeric, that number will be the coefficient
            - If the string contains non-numeric characters, then the only non-numeric characters that can be there are:
                - variables in self.variables
                - constants
                - integers (0-9)
                    - Note that '10' will be interpreted as 1 * 0. If you would like to use any multi-digit integer, you can include it in 'operator'
                - functions in self.functions
                    - Currently, vector-valued functions cannnot be coefficients
                - 'π' or 'e'
                - '^' or '/'
                - \* If an integer happens after a function name, then it will be assumed you are indexing a vector valued function as opposed to multiplying by that numeric character
                    - That is, 'u1' will be interpreted as u being a vector-valued function and 1 being its first component as opposed to u * 1
            - where the respective thing will be multiplied if it is a value
            - If it is an operator (^ or /), then that will change the current setting to match that of the operator
                - All future elements in the string will be applied in adherence to the current setting
                - The default is multiplication, and it will be reset to multiplication if you add a space ' ' in the string
                - For instance, 'xz^2y/3u' would be ((xz)^(2y))/(3u) while 'x z^2 y/3 u' would be x\*(z^2)\*(y/3)\*u
                - For reasons, if '-' was included at the beginning of the string, it will be applied after the calculation of the coefficient so something like -x^2 will be read as -(x^2)
        - For vector coefficients:
            - The string must contain symbols '(' and ')' or '[' and ']', which enclose the elements of the vector. The elements should be separated by commas ','
            - Each element will be parsed like a scalar coefficient
            - The string can also contain things to the left or right of the brackets, which will be multiplied element-wise into the string
    - If it was given of the form equation_structure, then the coefficient will be the given equation
    - If this key is not included, the default will be 1
    - Ex. {'coef':'2π',...}, {'coef':np.pi,...}, {'coef':'2xtu^2',...}, {'coef':'3(xt^2,u)'}, or:
    ```python
    {
        ### Coefficient is sin(x) + cos(y)
        'coefficient':[
            {'var':'x',
             'operator': lambda x: ko.sin(x)},
            {'var':'y',
             'operator': lambda y: ko.cos(y)},
        ],
        ...
    }
    ```
- KEY: 'operator' or 'op' (opt | default=lambda x: x)
    - The value should either be a callable or a string
    - This will be the operator that acts on the focus
    - If a callable, it should only accept one parameter and the focus will be passed as the argument. Only use tensorflow, keras.ops, or standard arithmetic in making this operator to ensure gradients work
        - Do not use things like np.sin
    - If a string, it should be one of:
        - 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'ln'
        - Where that operator will then be acting on the focus
        - If the focus is a vector-valued function, the operator will be applied to every component
    - If this key is not included, the default will be the identity function (lambda x: x)
    - Ex. {'operator': lambda y: ko.sin(np.pi*y)}
- KEY: 'apply_coef' or 'apply_coefficient' (opt | default='after')
    - The value should be a string
    - If 'before', the coefficient will be applied before the operator
    - If 'after', the coefficient will be applied after the operator

## If string

CURRENTLY CANNOT SUPPORT NABLA (div, curl, grad)

equation_structure can also be a string, where you get to write out the equation. However, this must still follow a particular format to be read properly by the parser.
- Each term must be separated by either a + or a -
    - You can also start with first term with a '-' to make it negative
- Do not use '=' anywhere. The equation is set to zero automatically
- For each term, the coefficients must be between round brackets
    - Powers (^) and division (/) are now supported in this
    - Please use the square brackets '[' and ']' if you want to use vector coefficients
- The focus of each term must be after the coefficient. This focus can only be of one variable, function, or component of a vector valued function fi
- In the focus, derivatives are indicated by the '_' symbol, where each character after that symbol will be a derivative
- You can also use operators in the focus. The operator must use () around its operand
    - If the coefficient to your term is 1 and you want to include an operator, you must still write "()" before your term
    - Currently, only string operators are supported

To see how it will be read by the parser, you can import and run tools.building.quick_parser.parse_eqn on your string.


## Examples

### Laplace equation

```python
equation_structure = [
    {
        "var": "u",
        "derivatives": ["x","x"],
        "coef": 1
    },
    {
        "var": "u",
        "derivatives": ["y","y"],
        "coef": 1
    }
]

### OR ###

equation_structure = "u_xx + u_yy"
```

### Forced wave equation

```python
equation_structure = [
    {
        "derivatives":['t','t'],
    },
    {
        "derivatives":['x','x'],
        "coef":"-c^2" # where c is included in constants. See below
    },
    {
        "var":"x",
        "op":lambda x: ko.cos(x)
    }
]

### OR ###

equation_structure = "u_tt - (c^2)u_xx - ()cos(x)"
```

### Maxwell's equations

```python
equation_structure = [

    # ∇ · E = 0
    [
        {
            "var": "E",
            "derivatives": ["∇(x,y,z)⋅"]
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

    # ∇ × B - ∂E/∂t = 0
    [
        {
            "var": "B",
            "derivatives": ["∇(x,y,z)x"]
        },
        {
            "var": "E",
            "derivatives": ["t"],
            "coef": -1
        }
    ]
]
```

# For conditions

conditions must be a list or tuple of dictionaries. The nth dictionary in this list denotes the nth condition.

Each dictionary in this list can have keys and values:
- KEY: 'equation', 'eqn' (req)
    - The value should be a list, tuple, or string of the same format as equation_structure
    - This specifies the equation of the condition
    - Remember to put all terms on one side when thinking of the equation
    - Ex. The following equation would be for u(x_location,y)=sinh(pi)\*sin(pi\*y)
    ```python
    {
        'equation':[
            {
                "var":"u",
                "coef":1
            },
            {
                "var":"y",
                "operator": lambda y: -ko.sinh(np.pi)*ko.sin(np.pi*y)
            }
        ],
        ...
    }
    ```
- KEY: 'location' or 'loc' (req)
    - The value should be a dictionary with keys being the variable and values being some value within the bounds of that variable
    - This specifies where the condition lies
    - Ex.
        - {'location':{'t':0},...} - This specifies an initial condition
        - {'location':{'t':0,'x':0},...} - This specifies a condition at (x,t) = (0,0)
    - ** This cannot handle locations like x = y, x^2 + y^2 = 1, etc. If you would like to use those locations, you can change your variables **
- KEY: 'n_samples', 'n-samples', or 'samples' (opt | default=50)
    - The value should be an integer
    - This specifies the number of samples to draw uniformly between the bounds for every other variable that wasn't fixed by the location
    - The resulting array passed into the equation will be of shape (n_samples,n_variables)
    - If this key is not included, the default will be 50
    - Ex. {'n_samples':50,...}

## Examples

```python
conditions = [

    # x = 0
    # u(0,y)=0
    {
        "location": {"x":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },


    # x = 1
    # u(1,y)=sinh(pi)*sin(pi*y)
    {
        "location": {"x":1},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            },
            {
                "var":"y",
                "operator": lambda y: -ko.sinh(np.pi)*ko.sin(np.pi*y)
            }
        ]
    },


    # y = 0
    # u(x,0)=0
    {
        "location":{"y":0},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    },


    # y = 1
    # u(x,1)=0
    {
        "location":{"y":1},
        "n_samples":100,
        "equation":[
            {
                "var":"u",
                "coef":1
            }
        ]
    }

]
```

## FOR COUPLED EQUATIONS

Nothing much changes here, except now you can specify multiple equations.

You can simply write equation_structure as a list or tuple of list or tuples, where each inner list or tuple is of the form model_structure above.

You can also write it as a list of strings, if you'd like to use the string parser.

## Examples

```python
equation_structure = ["u_x - v",
                      "v_x + u"]
```

# For constants

constants must be a list or tuple of dictionaries. The nth dictionary denotes the nth constant

Each dictionary in this list can have keys and values:
- KEY: 'name' (req)
    - The value should be a one character string which is the name of the constant
    - This specifies the constant's name and will be how you refer to it when using it as a coefficient
    - Ex. {'name':'c',...}
- KEY: 'value' or 'val' (req)
    - The value should be of type Number
    - This specifies the value of the constant. If it is trainable, this will be the initial value
    - Ex. {'value':3,...}
- KEY: 'trainable' or 'train' (opt | default=False)
    - The value should be a bool
    - This specifies whether the constant is trainable by gradient descent
    - This feature is added so inverse PINNs are supported
    - Ex. {'trainable':False,...}
- KEY: 'dtype' or 'type' (opt | default='float32')
    - The value should be a string
    - This specifies the type of the constant
    - Ex. {'dtype':'float64'}

## Examples

```python
constants = [
    {'name':'c',
     'value':3,
     'trainable':False,
     'dtype':'float64'},
    
    {'name':'d',
     'value':1,
     'trainable':True,
     'dtype':'float32'}
]
```

Then, in equation_structure, you can do:

```python
{'coef':'3c',...}
```

If you just want to add or subtract the constant, make the focus of the term 'const' and set the coefficient to be 'c'.