import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.random as kr
import tensorflow.keras.ops as ko
import numpy as np
from numbers import Number
import matplotlib.pyplot as plt

from .estimator import DeepEstimator

from .tools.base.pinn_base import PINN
from .tools.score import compute_score
from .tools.validation import validate_structure
from .tools.building.struct_tools import get_any
from .tools.building.quick_parser import parse_eqn

class DeepPINN(DeepEstimator):

    scoring_func = staticmethod(lambda _,residual: -residual)
    must_be_vector = True

    def __init__(self,
                 variables,
                 equation_structure,
                 conditions,
                 bounds,
                 n_samples,
                 constants=None,
                 data=None,
                 functions=None,
                 loss_weighting=None,
                 **kwargs):
        '''
        Parameters
        ----------
        variables : list or tuple
            A list of variables.
            
            Each can only be one character.
            
            Ex. ``['x','y','z']``
        
        equation_structure : list or tuple or string
            Similar to model_structure but species the equation.

            See ``equation.md`` on how to format this.

        conditions : list or tuple
            Similar to model_structure but specifies the conditions.

            See ``equation.md`` on how to format this.
        
        bounds : dict
            The bounds for each variable with keys being the variables and 
            values being a two-element tuple (lo,hi) or list [lo,hi]
            
            For example::

                {'x':(0,1),
                 'y':(0,1),
                 'z':(0,1)}
            
        n_samples : int
            The number of samples to be given to the main PDE. 
            The shape will be ``(n_samples,len(variables))``

            Numbers will be uniformly sampled between the bounds for the variable.
        
        constants : list or tuple or None, default=None
            Similar to model_structure but specifies the constants in the equation.

            If None, there will be no constants.

            See ``equation.md`` on how to format this.

        data : tf.Tensor or None, default=None
            A tensor of shape ``(n_samples,len(variables) + 1)``
            
            The first ``len(variables)`` columns contain the coordinates of each 
            sample point (in the same order as ``variables``), while the final column 
            contains the corresponding observed value of the solution.

            Often used in tandem with trainable constants for iPINNs.

            If None, there will be no extra data.

            Currently, this must be float32.
        
        functions : list or None, default=None
            A list of function names to be used in the equation.

            Each can only be one character.

            Ex. ``['u','v','w']``

            If None, only the default function 'u' will be used.

            Must be specified for vector-valued outputs or coupled equations.

        loss_weighting : dict or None, default=None
            The weighting for each loss.
            
            If given, must be of the form::

                {'pde':x,
                 'conditions':y, # or 'conds', 'cond'
                 'data':z}
            
            Where the loss will be calculated by L = x*L_pde + y*L_cond + z*L_data.

            ``'data'`` is not necessary if there was none given.

            If None, no loss weighting will be applied.
            
        **kwargs
            Inherited from DeepEstimator
        '''
        super().__init__(**kwargs)

        self.variables = variables
        self.equation_structure = equation_structure
        self.conditions = conditions
        self.bounds = bounds
        self.n_samples = n_samples
        self.constants = constants
        self.data = data
        self.functions = functions
        self.loss_weighting = loss_weighting

    def _validate_hyperparams(self):
        '''
        Validates the hyperparameters of the model.

        Will raise an error of they are not proper.

        Raises
        ------
        TypeError
            If model structure is not a list or tuple. 

            If any element in model structure is not a dictionary. 

            If equation structure is not a list or tuple.

            If conditions is not a list or tuple.

            If any element in conditions is not a dictionary.

            If bounds is not a dict.

            If constants is not a list or tuple.

            If loss_weighting is not a dictionary.
        
        ValueError
            If model structure is empty.
            
            If equation structure is empty.

            If variables is empty.

            If functions is empty.

            If validation split is not in [0,1). 

            If early stopping is True and validation split <= 0.
        
        KeyError
            If any element in model structure does not have key 'type'.

            If loss_weighting does not have keys 'pde' and 'conditions' and, if applicable, 'data'.
        '''
        super()._validate_hyperparams()

        validate_structure(self.equation_structure,"equation_structure",build_setting="not normal")
        validate_structure(self.conditions,"conditions",can_be_empty=True)
        validate_structure(self.constants,"constants",can_be_empty=True)

        if any(isinstance(eqn,(list,tuple)) for eqn in self.equation_structure) and \
           any(not isinstance(eqn,(list,tuple)) for eqn in self.equation_structure):
            raise ValueError("equation_structure must contain only lists/tuples or only dicts")

        if not isinstance(self.bounds,dict):
            raise TypeError("bounds must be a dictionary")

        if len(self.variables) == 0:
            raise ValueError("variables cannot be empty!")

        if len(self.functions) == 0:
            raise ValueError("functions cannot be empty!")

        if not isinstance(self.loss_weighting,dict):
            raise TypeError("loss_weighting must be a dictionary!")

        if 'pde' not in self.loss_weighting or 'conditions' not in self.loss_weighting or ('data' not in self.loss_weighting and self.data is not None):
            raise KeyError("loss_weighting must have keys 'pde', 'conditions', and 'data' (if data was given)")

    ### CALCULATING EQUATIONS AND CONDITIONS ###

    def _calc_eqn(self,X,structure):
        '''
        Parameters
        ----------
        X : np.darray
            An array of shape (n_samples,n_vars)
        
        structure : list, tuple, or None, default=None
            Specifies the structure of the equation.
            
            Of the form equation_structure.

        Returns
        -------
        result : tf.Tensor
            The result of the equation.
        '''

        if not tf.is_tensor(X):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        variables = self.variables
        functions = self.functions
        var_to_val = {}
        func_to_val = {}
        derivs = {}

        with tf.GradientTape(persistent=True) as tape:
            for ind,v in enumerate(variables):
                var_to_val[v] = X[:,ind:ind+1]
                tape.watch(var_to_val[v])

            eval = self.model_(ko.stack([val[:,0] for val in var_to_val.values()],axis=1))
            for ind,f in enumerate(functions):
                func_to_val[f] = eval[:,ind:ind+1]

            for ind, struct in enumerate(structure):
                var = get_any(struct,['variable','var'],fallback=functions[0])
                derivatives = get_any(struct,['derivatives','deriv'],fallback=[])

                if var not in functions:
                    if len(derivatives) != 0:
                        raise ValueError("Derivatives can only operate on functions")
                    continue

                derivs[ind] = func_to_val[var]
                
                for i,d in enumerate(derivatives[:-1]):
                    i += 1
    
                    val = var_to_val.get(d)

                    if val is None:
                        raise ValueError(f"Derivative {d} is not a variable in the given variables")
                    
                    grad = tape.gradient(derivs[ind],val)
    
                    if i % 10 == 1 and i % 100 != 11:
                        i = f"{i}st"
                    elif i % 10 == 2 and i % 100 != 12:
                        i = f"{i}nd"
                    elif i % 10 == 3 and i % 100 != 13:
                        i = f"{i}rd"
                    else:
                        i = f"{i}th"
    
                    if grad is None:
                        raise RuntimeError(f"Could not compute {i} derivative with respect to {d}")
                    
                    derivs[ind] = grad

        const = ko.ones_like(X[:,0:1])

        result = 0

        for ind,struct in enumerate(structure):

            var = get_any(struct,['variable','var'],fallback=functions[0])
            derivatives = get_any(struct,['derivatives','deriv'],fallback=[])
            coef = get_any(struct,['coefficient','coef'],fallback=1)
            operator = get_any(struct,['op','operator'],fallback=lambda x: x)

            if len(derivatives) != 0:
                if var not in functions:
                    raise ValueError("Derivatives can only operate on functions")
                
                d = derivatives[-1]
                grad = tape.gradient(derivs[ind],var_to_val[d])

                if grad is None:
                    raise RuntimeError(f"Could not compute last derivative with respect to {d}")

                derivs[ind] = grad

            if isinstance(coef,Number):
                cfs = coef
            elif isinstance(coef,str):
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
                    for char in coef:
                        to_apply = None
                        op_change = False

                        if char in functions:
                            to_apply = func_to_val[char]
                        elif char in variables:
                            to_apply = var_to_val[char]
                        elif char in self.constants_:
                            to_apply = self.constants_[char]
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
                        
            elif isinstance(coef,(list,tuple)):
                cfs = self._calc_eqn(X,coef)
            else:
                raise ValueError(f"Unknown coefficient type {type(coef)}")

            if var in functions:
                var_val = derivs[ind]
            elif var == 'const':
                var_val = const
            elif var in variables:
                var_val = var_to_val[var]
            else:
                raise ValueError(f"Unknown variable {var}")

            str_to_op = {
                'identity':lambda x: x,
                'sin':ko.sin,
                'sinh':ko.sinh,
                'cos':ko.cos,
                'cosh':ko.cosh,
                'tan':ko.tan,
                'tanh':ko.tanh,
                'ln':ko.log
            }

            if isinstance(operator,str):
                if operator not in str_to_op:
                    raise ValueError(
                        f"If string, operator must be in {list(str_to_op.keys())}"
                    )

                name = operator
                operator = lambda var: str_to_op[name](var)

            result += operator(var_val)*cfs

        del tape

        return result

    def _calc_conds(self,X_b):
        '''
        Parameters
        ----------
        X_b : list
            A list of arrays containing the condition data.
            
            The ith element denotes 
            the array for the ith condition
        
        ind : int
            The index at which to calculate the condition for.

        Returns
        -------
        result : list
            A list of tf.Tensors, each representing the result of a condition in the condition structure.

            The nth element of the list corresponds to the nth condition in the condition structure.
        '''
        result = []
        for i in range(len(X_b)):
            result.append(self._calc_eqn(X_b[i],
                                     get_any(self.conditions[i],
                                             ['eqn','equation']
                                         )
                            ))
        return result

    def _calc_pde(self,X_r):
        '''
        Parameters
        ----------
        X_r : array-like
            An array of shape (n_samples,n_vars)
        
        Returns
        -------
        result : list
            A list of tf.Tensors, each representing the result of a PDE in the equation structure.

            The nth element of the list corresponds to the nth PDE in the equation structure.
        '''
        eqn_structure = self.equation_structure
        if all(isinstance(eqn,(list,tuple)) for eqn in eqn_structure):
            result = []
            for eqn in eqn_structure:
                result.append(self._calc_eqn(X_r,eqn))
        else:
            result = [self._calc_eqn(X_r,eqn_structure)]

        return result

    ### PREPARATION BEFORE FITTING ###
    
    def _get_data(self,loc,n_samples,mins,maxs,label=None):
        '''
        Gets the data for a given location, n_samples, min, max

        Parameters
        ----------
        loc : dict
            Fixed location.

        n_samples : int
            Number of samples to draw uniformly between mins and maxs.

        mins : array-like
            Minimums of variables.

        maxs : array-like
            Maximums of variables.

        label : anything
            The label before the error.
        
        Returns
        -------
        data : tf.Tensor
        '''
        variables = self.variables

        for var in loc:
            if var not in variables:
                raise ValueError(f"{label}{var} is not one of the given variables")

            if not (self.bounds[var][0] <= loc[var] <= self.bounds[var][1]):
                raise ValueError(f"{label}location for {var} must be between the bounds!")

        return (ko.concatenate(
                            [ko.ones((n_samples,1))*loc[var] if v in loc
                            else kr.uniform((n_samples,1),mins[i],maxs[i])
                            for i,v in enumerate(self.variables)],
                            axis=1
        ))

    def _prepare_hyperparams(self):
        '''
        Prepares the hyperparameters before training
        '''
        if self.functions is None:
            self.functions = ['u']

        if self.constants is None:
            self.constants = []

        if self.loss_weighting is None:
            self.loss_weighting = {'pde':1,'conditions':1,'data':1}

        if 'conds' in self.loss_weighting:
            self.loss_weighting['conditions'] = self.loss_weighting.pop('conds',1)
        elif 'cond' in self.loss_weighting:
            self.loss_weighting['conditions'] = self.loss_weighting.pop('cond',1)

        if isinstance(self.equation_structure,str):
            self.equation_structure = parse_eqn(self.equation_structure)

        for ind,eqn in enumerate(self.equation_structure):
            if isinstance(eqn,str):
                self.equation_structure[ind] = parse_eqn(eqn)

        for ind,cond in enumerate(self.conditions):
            eqn = get_any(cond,
                          ['eqn','equation'],
                          err=KeyError(
                                f"No equation given for condition {ind}"
                                ))
            if isinstance(eqn,str):
                cond['eqn'] = parse_eqn(eqn)

    def _prepare_data(self):
        '''
        Prepares the data for training
        '''
        variables = self.variables

        ### Get bounds ###
        mins = []
        maxs = []
        for v in variables:
            b = self.bounds[v]
            mins.append(b[0])
            maxs.append(b[1])

        self.mins = mins
        self.maxs = maxs

        ### Preparation for PDE ###
        X_r = self._get_data({},self.n_samples,mins,maxs)

        ### Preparation for conditions ###
        conds = self.conditions

        X_b_data = []

        for ind, structure in enumerate(conds):
            loc = get_any(structure,['loc','location'],err=KeyError(f"No location given for condition {ind}"))
            n_samples = get_any(structure,['n_samples','n-samples','samples'],50)

            X_b_data.append(self._get_data(loc,n_samples,mins,maxs,f"Condition {ind}: "))

        self.X_r = X_r
        self.X_b_data = X_b_data

    def _prepare_constants(self):
        self.constants_ = {}

        for ind,c in enumerate(self.constants):
            name = get_any(c,['name'],err=f"No name given for constant {ind}")

            if not isinstance(name,str):
                raise ValueError(f"Name for constant {ind} must be a string. Type of name given: {type(name)}")
            if len(name) != 1:
                raise ValueError(f"Name for constant {ind} must be single character. Name given: {name}")

            if name in self.variables:
                raise ValueError(f"Name for constant {name} is also the name of a variable.")

            value = get_any(c,['val','value'],err=f"No value given for constant {ind}")
            trainable = get_any(c,['trainable','train'],False)
            dtype = get_any(c,['dtype','type'],'float32')

            self.constants_[name] = keras.Variable(value,dtype=dtype,trainable=trainable)


    ### SKLEARN METHODS ###

    def fit(self,X=None,y=None,**fit_params):
        '''
        Trains the model to predict the given PDE

        Parameters
        ----------
        X : array-like or None, default=None
            An array of shape ``(n_samples,n_variables)``.

            If None, one will be made from ``n_samples`` and ``bounds``.

            It is recommended to leave this as None for most cases.
        
        y : None, default=None
            Leave this as None
        
        **fit_params
            Any additional fit parameters used in Keras.
        
        Returns
        -------
        self
            The trained estimator.
        '''
        self._prepare_hyperparams()
        self._prepare_constants()
        self._prepare_data()

        self.y_was_1d_ = False
        self.is_multi_input_ = self.is_multi_output_ = False

        X = self.X_r if X is None else X

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        X = np.asarray(X)
        expec_inp = X.shape[1:]

        self.input_shape_ = self.input_shape if self.input_shape is not None else expec_inp

        structs = self._prepare_structure()

        self.model_ = PINN(self.X_b_data,
                           mins=self.mins,
                           maxs=self.maxs,
                           model=self._build_model(structs),
                           calc_pde=self._calc_pde,
                           calc_bound_eqn=self._calc_conds,
                           constants=self.constants_,
                           data=self.data,
                           loss_weighting=self.loss_weighting)
        self.model_.compile(
            optimizer=self._make_optimizer()
        )

        if len(self.output_shape_) >= 2:
            raise ValueError(f"Output shape must be 1D. Output shape given: {self.output_shape_}")
        
        if self.output_shape_[0] != len(self.functions):
            raise ValueError(f"Output shape must be equal to the number of functions. "
                             f"Output shape given: {self.output_shape_}, "
                             f"number of functions given: {len(self.functions)}")

        X = self._validate_data(X)

        callbacks = self._get_callbacks()
        history = self.model_.fit(
                    X,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=self.verbose,
                    shuffle=self.shuffle,
                    **fit_params,
                )

        self.history_ = history.history
        self.loss_curve_ = history.history.get("loss")
        self.validation_scores_ = history.history.get("val_loss")

        return self

    def predict(self,X=None):
        '''
        Predicts the values of the function given the input.

        Parameters
        ----------
        X : array-like or None, default=None
            An array of shape ``(n_samples,n_variables)``.

            If None, one will be made from ``n_samples`` and ``bounds``.

        Returns
        -------
        y : numpy.ndarray
            The function evaluated at every point.
            
            Shape is ``(n_samples,n_outputs)``.
        
        Raises
        ------
        ValueError
            If the dimension of the features is not equal 
            to the dimension of the input.

            For multi-input, if the number of inputs is not 
            equal to the number of features

            For multi-input, if the number of samples is not 
            equal between arrays
        '''
        X_r = self.X_r if X is None else X
        return super().predict(X_r)

    def score(self,X=None,y=None):
        '''
        Scores how well the model performs on the PDE.

        Parameters
        ----------
        X : array-like or None, default=None
            An array of shape ``(n_samples,n_variables)``.

            If None, one will be made from ``n_samples`` and ``bounds``.
        
        y : None, default=None
            Leave this as None
        
        Returns
        -------
        score : np.float32
            The negative loss
        '''
        self._check_is_fitted()
        X_r = self.X_r if X is None else X
        return compute_score(None,
                             self.model_.get_loss(X_r).numpy(),
                             self.scoring_func,
                             weights=self.scoring_weights,
                             must_be_vector=self.must_be_vector)


    ### CUSTOM METHODS ###

    def predict_at_loc(self,loc,n_samples=50):
        '''
        Predicts the value of the function at a given location.

        Parameters
        ----------
        loc : dict
            The location to evaluate the function of the form::

                {var1:val1,
                 var2,val2,
                 ...}
            
            All variables not in the keys will be considered unfixed.
        
        n_samples : int, default=50
            The number of samples to draw uniformly between the bounds for the unfixed variables.
        
        Returns
        -------
        pred : numpy.ndarray
            The function evaluated at every point.

            Shape is ``(n_samples,n_outputs)``
        '''
        self._check_is_fitted()
        return self.predict(self._get_data(loc,n_samples,self.mins,self.maxs,label="Running Predict_at_loc: "))

    def plot(self,
             loc,
             n_samples=50,
             draw=True,
             ax=None):
        '''
        Plots the function.

        For one free variable, generates a u vs. free_var plot.

        For two free variables, generates a colored scatter plot.

        Cannot handle any other amount of free variables.

        Parameters
        ----------
        loc : dict
            The location to evaluate the function of the form::

                {var1:val1,
                 var2,val2,
                 ...}
            
            All variables not in the keys will be considered unfixed.

            The number of unfixed variables can be no more than 2.
        
        n_samples : int, default=50
            The number of samples to draw uniformly between the bounds for the unfixed variables.
        
        draw : bool, default=True
            If True, will call plt.show()
        
        ax : matplotlib.axes.Axes or None, default=None
            The axes object to plot with.

            If None, one will be created.
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib.axes.Axes object belonging 
            to the plotted distribution
        '''
        self._check_is_fitted()

        X = self._get_data(loc,n_samples,self.mins,self.maxs)
        pred = self.predict(X)

        free_vars = [(i,v) for i,v in enumerate(self.variables) if v not in loc]

        if ax is None:
            _,ax = plt.subplots()

        if len(free_vars) == 1:
            free_ind,free_var = free_vars[0]
            x = np.asarray(X[:,free_ind])
            y = np.asarray(pred[:,0])

            order = np.argsort(x)
            ax.plot(x[order],y[order])
            ax.set_xlabel(free_var)
            ax.set_ylabel("u")

        elif len(free_vars) == 2:
            x = X[:,free_vars[0][0]]
            y = X[:,free_vars[1][0]]

            ax.scatter(x,y,c=pred[:,0],label='u')
            ax.set_xlabel(free_vars[0][1])
            ax.set_ylabel(free_vars[1][1])
            ax.legend()

        else:
            raise ValueError(f"Unable to plot {len(free_vars)} free variables")

        if draw:
            plt.show()

        return ax
