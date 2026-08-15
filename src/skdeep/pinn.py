import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.random as kr
import tensorflow.keras.ops as ko
import numpy as np
from numbers import Number
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .estimator import DeepEstimator

from .tools.base.pinn_base import PINN
from .tools.score import compute_score
from .tools.validation import validate_structure
from .tools.building.struct_tools import get_any
from .tools.building.quick_parser import parse_eqn
from .tools.equation.deriv import find_deriv
from .tools.equation.coeff import parse_scalar_coef,parse_vector_coef

class DeepPINN(DeepEstimator):

    scoring_func = staticmethod(lambda _,residual: -residual)
    must_be_vector = True

    def __init__(self,
                 variables,
                 equation_structure,
                 conditions,
                 bounds,
                 n_samples,
                 functions=None,
                 constants=None,
                 data=None,
                 coordinates='cartesian',
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
        
        functions : list or None, default=None
            A list of function names to be used in the equation.

            Each can only be one character.

            Ex. ``['u','v','w']``

            If None, only the default function 'u' will be used.

            Must be specified for vector-valued outputs or coupled equations.
        
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
        
        coordinates : string, default='cartesian'
            Specifies the coordinate system.

            Must be either 'cartesian', 'polar', 'cylindrical', or 'spherical'.

            The only thing this changes is the behavior of the multivariate derivatives.

            The number of variables between the brackets ``[...]`` 
            when applying the multivariate derivative must be:

            2 when ``coordinates = 'polar'`` and be of the order ``[r,θ]``. 
            The vector field to act on or is produced should be of order F_1 = F_r and F_2 = F_θ.

            3 when ``coordinates = 'cylindrical'`` and be of the order ``[r,azimuth,z]``. 
            The vector field to act on or is produced should be of order F_1 = F_r, F_2 = F_azimuth, and F_3 = F_z.

            3 when ``coordinates = 'spherical'`` and be of the order ``[ρ,polar,azimuth]``. 
            The vector field to act on or is produced should be of order F_1 = R_ρ, F_2 = F_polar, F_3 = F_azimuth.

            Any amount when ``coordinates = 'cartesian'``.

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
        self.functions = functions
        self.constants = constants
        self.data = data
        self.coordinates = coordinates
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

            If any constant name is not a string
        
        ValueError
            If model structure is empty.
            
            If equation structure is empty.

            If variables is empty.

            If functions is empty.

            If validation split is not in [0,1). 

            If early stopping is True and validation split <= 0.

            If any constant uses only numeric characters in the name.

            If the length of any constant name is greater than 1.

            If any banned name is used in a constant's name, variable's name, or function's name
        
        KeyError
            If any element in model structure does not have key 'type'.

            If loss_weighting does not have keys 'pde' and 'conditions' and, if applicable, 'data'.
        '''
        super()._validate_hyperparams()

        validate_structure(self.equation_structure,"equation_structure",build_setting="not normal")
        validate_structure(self.conditions,"conditions",can_be_empty=True)
        validate_structure(self.constants,"constants",can_be_empty=True)

        if any(not isinstance(name,str) for name in self.constants_):
            raise TypeError("All constant names must be strings")

        if any(name.isnumeric() for name in self.constants_):
            raise ValueError("Constant names cannot be numbers!")

        if any(len(name) != 1 for name in self.constants_):
            raise ValueError("Constant names can only be one character long!")

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

        banned_names = ['(',')',';',',','\\','.','?','!','#','@',
                        '$','%','^','&','*','-','=','+','/','[',']',
                        '>','<','{','}','_']
        
        for bn in banned_names:
            if any(bn in name for name in self.constants_):
                raise ValueError(f"{bn} cannot be used in a constant's name")

            if any(bn in name for name in self.variables):
                raise ValueError(f"{bn} cannot be used in a variable's name")

            if any(bn in name for name in self.functions):
                raise ValueError(f"{bn} cannot be used in a function's name")

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

                if var_to_val[v].shape[1] == 0:
                    raise ValueError(f"X is not of the right shape. Expected shape: (n_samples,{len(variables)}). "
                                     f"Given shape: {X.shape}")
                
                tape.watch(var_to_val[v])

            ### For each function, eval is of shape (n_samples,n_outputs)
            eval = self.model_(ko.stack([val[:,0] for val in var_to_val.values()],axis=1))

            for ind,f in enumerate(functions):

                # If there is no multi-headed output, then ev is just eval
                if not self.is_multi_output_:
                    ev = eval

                # If there is multi-headed output, then eval is a list
                else:
                    ev = eval[ind]

                # If the current function is vector valued, we need to assign fi to ev[:,i-1:i]
                # i = 1,2,3... is numbering
                if ev.shape[1] >= 2:
                    for i in range(1,ev.shape[1]+1):
                        func_to_val[f"{f}{i}"] = ev[:,i-1:i]

                    func_to_val[f] = ev

                # If the current function is scalar valued, we can just assign f to ev
                else:
                    func_to_val[f] = ev

            for ind, struct in enumerate(structure):
                var = get_any(struct,['variable','var'],fallback=functions[0])
                derivatives = get_any(struct,['derivatives','deriv'],fallback=[])

                if var[0] not in functions:
                    if len(derivatives) != 0:
                        raise ValueError("Derivatives can only operate on functions")
                    continue

                derivs[ind] = func_to_val.get(var)
                if derivs[ind] is None:
                    raise ValueError(f"{var} is not a function in the given functions")

                for i,d in enumerate(derivatives):
                    find_deriv(i,d,var_to_val,tape,variables,derivs,ind,self.coordinates)

        result = 0

        for ind,struct in enumerate(structure):

            var = get_any(struct,['variable','var'],fallback=functions[0])
            coef = get_any(struct,['coefficient','coef'],fallback=1)
            operator = get_any(struct,['op','operator'],fallback=lambda x: x)
            apply_when = get_any(struct,['apply_coef','apply_coefficient'],fallback='after')

            if isinstance(coef,Number):
                cfs = coef
            elif isinstance(coef,str):
                parser = parse_vector_coef if ',' in coef else parse_scalar_coef
                cfs = parser(coef,var_to_val,func_to_val,self.constants_)
            elif isinstance(coef,(list,tuple)):
                cfs = self._calc_eqn(X,coef)
            else:
                raise ValueError(f"Unknown coefficient type {type(coef)}")

            if var[0] in functions:
                var_val = derivs[ind]
            elif var == 'const':
                var_val = ko.ones_like(result)
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

            if tf.is_tensor(result):
                if result.shape != var_val.shape:
                    raise ValueError(f"All terms in the equation must have the same shape. "
                                     f"Shape of result: {result.shape}. "
                                     f"Shape of term {ind}: {var_val.shape}")

            if apply_when.lower() == 'before':
                res = operator(cfs*var_val)
            elif apply_when.lower() == 'after':
                res = operator(var_val)*cfs
            else:
                raise ValueError("apply_when must either be 'before' or 'after'")
            
            result += res

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
    
    def _get_data(self,loc,n_samples,mins,maxs,label=None,random=True,concat=True):
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
            The label used in the error.
        
        random : bool, default=True
            If True: samples randomly.

            If False: samples using fixed intervals.
        
        concat : bool, default=True
            Whether to concatenate the data before returning

        Returns
        -------
        data : tf.Tensor or List[tf.Tensor]
        '''
        variables = self.variables

        for var in loc:
            if var not in variables:
                raise ValueError(f"{label}{var} is not one of the given variables")

            if not (self.bounds[var][0] <= loc[var] <= self.bounds[var][1]):
                raise ValueError(f"{label}location for {var} must be between the bounds!")

        if random:
            data = [ko.ones((n_samples,1))*loc[var] if v in loc
                    else kr.uniform((n_samples,1),mins[i],maxs[i])
                    for i,v in enumerate(self.variables)]
        else:
            data = [np.ones((n_samples,1))*loc[var] if v in loc
                    else np.linspace(mins[i],maxs[i],n_samples)
                    for i,v in enumerate(self.variables)]

        if concat: data = ko.concatenate(data,axis=1)

        return data

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
        self._prepare_data()

        self.y_was_1d_ = False
        self.is_multi_input_ = False

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

        if len(self.functions) != 1:
            if not self.is_multi_output_:
                raise ValueError(f"Multi-headed output needed for multiple functions.")

            if len(self.functions) != len(self.output_shape_):
                raise ValueError(f"Number of functions must be equal to number of output heads. "
                                 f"Functions given: {len(self.functions)}. "
                                 f"Output heads given: {len(self.output_shape_)}")
        else:
            if self.is_multi_output_:
                raise ValueError(f"Multi-headed output not needed for single function.")

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

        self.y_was_1d_ = [False]*len(self.output_shape_) if self.is_multi_output_ else False

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
            
            Shape is ``(n_samples,n_outputs)`` for single function or a list of arrays for multiple functions.
        
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

    def predict_at_loc(self,
                       loc: dict,
                       n_samples: int = 50):
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

            Shape is ``(n_samples,n_outputs)`` for single function or a list of arrays for multiple functions.
        '''
        self._check_is_fitted()
        return self.predict(self._get_data(loc,n_samples,self.mins,self.maxs,label="Running Predict_at_loc: "))

    def plot(self,
             loc: dict,
             n_samples: int = 50,
             function: str | None = None,
             vec_el: int | tuple | None = None,
             draw: bool = True,
             ax: Axes | None = None,
             use_heatmap: bool = True) -> Axes:
        '''
        Plots the function.

        For one free variable, generates a u vs. free_var plot.

        For two free variables, generates a heatmap, surface, or vector field

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
        
        function : str or None, default=None
            The function to plot.
            
            If None, it will be ``self.functions[0]``.
        
        vec_el : int, tuple, or None, default=None
            For vector valued functions, the element of the vector to plot. 
            Counting from 1 (as opposed to 0).

            If None, the vector field will be drawn.

            If tuple, the corresponding components of the vector field will be drawn.

            Leave this as None if you are plotting a scalar function.
        
        draw : bool, default=True
            If True, will call plt.show()
        
        ax : matplotlib.axes.Axes or None, default=None
            The axes object to plot with.

            If None, one will be created.
        
        use_heatmap : bool, default=True
            Whether to plot a function from R^2 --> R using a heatmap.

            If False, the function will be plotted as a surface.
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib.axes.Axes object belonging 
            to the plotted distribution
        '''
        self._check_is_fitted()

        data = self._get_data(loc,
                              n_samples,
                              self.mins,
                              self.maxs,
                              label="Plotting: ",
                              random=False,
                              concat=False)

        free_vars = [(i,v) for i,v in enumerate(self.variables) if v not in loc]
        free_data = [data[i] for i,_ in free_vars]
        meshgrid = np.meshgrid(*free_data)

        X = np.zeros((np.prod(meshgrid[0].shape), len(self.variables)))

        for i, var in enumerate(self.variables):
            if var in loc:
                X[:, i] = loc[var]
            else:
                free_i = [j for j,(ind,_) in enumerate(free_vars) if ind == i][0]
                X[:, i] = meshgrid[free_i].ravel()

        pred = self.predict(X)

        f_ind = self.functions.index(function) if function is not None else 0
        f_label = function if function is not None else self.functions[0]

        if isinstance(pred,(list,tuple)):
            pred = pred[f_ind]

        if pred.shape[1] > 1:
            is_vec = True
        elif vec_el is None:
            vec_el = 1
            is_vec = False

        if isinstance(vec_el,tuple):
            if len(vec_el) == 1:
                vec_el = vec_el[0]
            elif len(vec_el) > pred.shape[1]:
                raise ValueError("Number of dimensions of vec_el is greater "
                                 "than the number of dimensions of the vector field")
            elif any(el > pred.shape[1] for el in vec_el):
                raise ValueError("vec_el number out of range")
            elif any(el == 0 for el in vec_el):
                raise ValueError("Counting starts from 1!")
        elif vec_el == 0:
            raise ValueError("Counting starts from 1!")

        if ax is None:
            fig = plt.figure()
            if (len(free_vars) == 2 and not use_heatmap) or len(free_vars) == 3:
                proj = '3d'
            else:
                proj = 'rectilinear'
            ax = fig.add_subplot(111,projection=proj)

        if len(free_vars) == 1:
            if is_vec and vec_el is None:
                raise ValueError("Cannot plot vector field for one free variable.")
            else:
                free_ind,free_var = free_vars[0]
                x = np.asarray(X[:,free_ind])
                y = np.asarray(pred[:,vec_el-1])

                order = np.argsort(x)
                ax.plot(x[order],y[order],label=f_label)
                ax.set_xlabel(free_var)
                ax.set_ylabel(f_label)
        elif len(free_vars) == 2:
            x_i, x_v = free_vars[0]
            y_i, y_v = free_vars[1]
            x_g, y_g = meshgrid

            if is_vec and (vec_el is None or isinstance(vec_el,tuple)):
                if vec_el is None:
                    vec_el = (1,2)

                if len(vec_el) != 2:
                    raise ValueError(f"Cannot plot {len(vec_el)}-D vector field in 2-D space. "
                                     "Vector field must be 2-D!")
                
                F1, F2 = pred[:,vec_el[0]-1],pred[:,vec_el[1]-1]
                ax.quiver(x_g.ravel(),y_g.ravel(),F1,F2,color='black',label=f_label)
            else:
                z = pred[:,vec_el-1].reshape(x_g.shape)
                if use_heatmap:
                    ax.pcolormesh(x_g,y_g,z,shading='auto',label=f_label)
                else:
                    if ax.name != '3d':
                        raise ValueError("axes given must be 3-D for plotting surfaces!")

                    ax.plot_surface(x_g,y_g,z)

            ax.set_xlabel(x_v)
            ax.set_ylabel(y_v)
        elif len(free_vars) == 3:
            x_i, x_v = free_vars[0]
            y_i, y_v = free_vars[1]
            z_i, z_v = free_vars[2]
            x_g, y_g, z_g = meshgrid

            if is_vec and (vec_el is None or isinstance(vec_el,tuple)):
                if vec_el is None:
                    vec_el = (1,2,3)
                
                if len(vec_el) != 3:
                    raise ValueError(f"Cannot plot {len(vec_el)}-D vector field in 3-D space. "
                                     "Vector field must be 3-D!")

                F1,F2,F3 = pred[:,vec_el[0]-1],pred[:,vec_el[1]-1],pred[:,vec_el[2]-1]
                ax.quiver(x_g.ravel(),y_g.ravel(),z_g.ravel(),F1,F2,F3,color='black',label=f_label)
            else:
                raise ValueError("Unable to plot scalar function in 4-D!")

            ax.set_xlabel(x_v)
            ax.set_ylabel(y_v)
            ax.set_zlabel(z_v)
        else:
            raise ValueError(f"Unable to plot {len(free_vars)} free variables")

        ax.legend()

        if draw:
            plt.show()

        return ax