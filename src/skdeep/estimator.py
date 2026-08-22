from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from scipy.sparse import issparse
from tensorflow import keras
from tensorflow.keras import layers as kl
import numpy as np
from typing import Literal,Callable

from .tools.building.quick_parser import parse_quick
from .tools.score import compute_score, neg_mse_score
from .tools.validation import validate_branches, validate_structure
from .tools.building.add_block import add_block
from .tools.types import ModelStructureType

class DeepEstimator(BaseEstimator):
    '''
    DeepEstimator is a machine learning algorithm that combines the user-friendly 
    features of scikit-learn regressors and the versatility of Tensorflow with Keras
    '''

    scoring_func = staticmethod(neg_mse_score)
    must_be_vector = False
    
    def __init__(
        self,
        model_structure: ModelStructureType,
        build_setting: Literal['normal','quick'] = "normal",
        input_shape: list | tuple | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        validation_split: float = 0.1,
        verbose: Literal[0,1,2,'auto'] = 1,
        loss: keras.losses.Loss | Callable | str | list = "mse",
        metrics: keras.metrics.Metric | list | tuple | dict | None = None,
        optimizer: keras.optimizers.Optimizer | str = "adam",
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        shuffle: bool = True,
        scoring_weights: list | tuple | None = None
    ):
        '''
        Parameters
        ----------
        model_structure : list or tuple
            Specifies the model architecture.

            See ``architecture.md`` for how to format this.

        build_setting : str, default="normal"
            Decides the format of model_structure.

            Must be either 'normal' or 'quick'.

            See ``architecture.md`` for more information.

        input_shape : tuple, list, or None, default=None
            The input shape.

            For single input, use a tuple specifying the input shape.

            For multi-input, use a list of tuples, where the ith tuple 
            denotes the input shape of the ith branch.

            If None, it will be guessed from the feature shape.

        epochs : int, default=100
            The number of epochs to train the model for.

        batch_size : int, default=32
            The batch size for training.

        early_stopping : bool, default=True
            Whether the model should stop training early if validation
            loss doesn't drop after n_iter_no_change iterations.

        n_iter_no_change : int, default=10
            The amount of iterations without validation loss change until
            the model stops training.

            (Only matters if early_stopping is True.)

        validation_split : float
            Should be between 0 and 1.

            This will determine how the training and validation data are split,
            with validation_split being the fraction of validation data.

        verbose : int
            If 0, nothing is printed.

            If 1, the process of training is printed.

        loss : keras.losses.Loss, str, callable, or list, default="mse"
            The loss function used. See Keras for custom ones.

            If your model has a multi-output layer, you can use a list
            where the ith loss corresponds to the ith output.

        metrics : keras.metrics.Metric, list, tuple, dict, or None, default=None
            The metrics tracked during training.

        optimizer : str or keras.optimizers.Optimizer, default="adam"
            The optimizer used in training.

            See Keras for possibilities.

        learning_rate : float, default=1e-4
            The learning rate for training.

        random_state : int or None, default=None
            The random state.

            Used for reproducible results.

        shuffle : bool, default=True
            Whether to shuffle the data before training.

        scoring_weights : list or tuple or None, default=None
            For multi-headed output only.

            Determines how the average score is weighted.

            The ith element of this denotes the weighting of the score
            corresponding to the ith output.
        '''
        self.model_structure = model_structure
        self.build_setting = build_setting
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_split = validation_split
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.shuffle = shuffle
        self.scoring_weights = scoring_weights


    ### VALIDATION ###

    def _validate_data(self,X,y=None):
        '''
        Checks the data to see if it is of the proper format.

        Parameters
        ----------
        X : array-like
            The feature array of shape ``(n,*input_shape_)`` for single input 
            or a list of features for multi-input.

        y : array-like or list of array-like, default=None
            The labels array of shape ``(n,*output_shape_)`` for single input 
            or a list of labels arrays for multi-output. 

            If None, only the features will be checked and returned.
        
        Returns
        -------
        X : np.ndarray or list of np.ndarray
            X formatted by sklearn's validate_data if single input
        
        y : np.ndarray or list of np.ndarray
            y formatted by sklearn's validate_data if single output

        Raises
        ------
        ValueError
            If the dimension of the labels is not equal 
            to the dimension of the output.

            If the dimension of the features is not equal 
            to the dimension of the input.

            For multi-output, if the number of outputs is not 
            equal to the amount of labels

            For multi-input, if the number of inputs is not 
            equal to the number of features

            For multi-input or output, if the number of samples 
            is not equal between arrays
        '''
        if self.is_multi_input_ or self.is_multi_output_:
            if self.is_multi_output_ and y is not None:
                if len(y) != len(self.output_shape_):
                    raise ValueError(
                        f"Expected {len(self.output_shape_)} outputs, "
                        f"got {len(y)} outputs instead"
                    )
                
                for i,(target,expec_shape) in enumerate(zip(y,self.output_shape_)):
                    if target.shape[1:] != expec_shape:
                        raise ValueError(
                            f"For output {i}: expected shape {expec_shape}, "
                            f"got {target.shape[1:]} instead"
                        )

                n_samples = y[0].shape[0]
                for ind,target in enumerate(y):
                    if target.shape[0] != n_samples:
                        raise ValueError(
                            f"The number of samples for label {ind+1}: {target.shape[0]}\n"
                            f"The number of samples for label 1: {n_samples}"
                        )

            if self.is_multi_input_:
                if len(X) != len(self.input_shape_):
                    raise ValueError(
                        f"Expected {len(self.input_shape_)} inputs, "
                        f"got {len(X)} inputs instead"
                    )

                for i,(inp,expec_shape) in enumerate(zip(X,self.input_shape_)):
                    if inp.shape[1:] != expec_shape:
                        raise ValueError(
                            f"For input {i}: expected shape {expec_shape}, "
                            f"got {inp.shape[1:]} instead"
                        )

                n_samples = X[0].shape[0]
                for ind,x in enumerate(X):
                    if x.shape[0] != n_samples:
                        raise ValueError(
                            f"The number of samples for feature {ind+1}: {x.shape[0]}\n"
                            f"The number of samples for feature 1: {n_samples}"
                        )
                    
        elif len(self.output_shape_) <= 1 and len(self.input_shape_) <= 1:
            if y is None:
                X = check_array(X, accept_sparse=False, dtype=np.float32)
                X = validate_data(self,X,reset=False)
            else:
                X, y = validate_data(
                    self,
                    X,
                    y,
                    multi_output=True,
                    y_numeric=True,
                    dtype=np.float32,
                )
        
        if y is not None and not self.is_multi_output_ and self.output_shape_ != y.shape[1:]:
            raise ValueError(
                f"output_shape={self.output_shape_}, but y has shape {y.shape[1:]}"
            )

        if not self.is_multi_input_ and self.input_shape_ != X.shape[1:]:
            raise ValueError(
                f"input_shape={self.input_shape_}, but X has shape {X.shape[1:]}"
            )

        return X if y is None else (X,y)

    def _validate_hyperparams(self):
        '''
        Validates the hyperparameters of the model.

        Will raise an error of they are not proper.

        Raises
        ------
        TypeError
            If model structure is not a list or tuple. 

            If any element in model structure is not a dictionary. 
        
        ValueError
            If model structure is empty. 

            If validation split is not in [0,1). 

            If early stopping is True and validation split <= 0.
        
        KeyError
            If any element in model structure does not have key 'type'.
        '''
        validate_structure(self.model_structure,"model_structure",self.build_setting)

        if self.build_setting == "normal":
            if any(struct.get('type') is None for struct in self.model_structure):
                raise KeyError("Each struct in model_structure must have key 'type'")
        
        if self.validation_split is not None and not (0 <= self.validation_split < 1):
            raise ValueError("validation_split must be in [0, 1).")
        
        if self.early_stopping and (self.validation_split is None or self.validation_split <= 0):
            raise ValueError("early_stopping=True requires validation_split > 0.")


    ### HELPER FUNCTIONS ###

    def _make_optimizer(self):
        '''
        Creates the optimizer for the model

        Returns
        -------
        Optimizer : keras.Optimizer
            The optimizer object
        '''
        if isinstance(self.optimizer, str):
            opt = keras.optimizers.get(self.optimizer)
        else:
            opt = keras.optimizers.deserialize(keras.optimizers.serialize(self.optimizer))

        if self.learning_rate is not None:
            opt.learning_rate = self.learning_rate

        return opt

    def _get_callbacks(self):
        '''
        Creates and returns a list of callbacks. 

        Currently only supports early stopping.

        Returns
        -------
        callbacks : list
            A list of callbacks.
        '''
        callbacks = []

        if self.early_stopping:
            patience = self.n_iter_no_change
            if patience is None:
                patience = max(5, int(self.epochs * 0.1))

            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        
        return callbacks

    def _format_data(self,X,y=None):
        '''
        Returns
        -------
        X : np.ndarray or list of np.ndarray
            X as an array for single input or 
            a list of arrays for multi-input
        
        y : np.ndarray or list of np.ndarray
            y as an array for single output or 
            a list of arrays for multi-output

            This is only returned if y is given.
        '''
        X = [np.asarray(x) for x in X] \
            if self.is_multi_input_ else np.asarray(X)

        if y is not None:
            y = [np.asarray(target) for target in y] \
                if self.is_multi_output_ else np.asarray(y)

        return X if y is None else (X,y)
    
    def _check_is_fitted(self):
        '''
        Checks if the model was fitted.
        '''
        check_is_fitted(self, "model_")

    def _prepare_structure(self):
        '''
        Prepares ``model_structure`` for model construction

        Returns
        -------
        structs : list
            The parsed model structure
        '''
        self._validate_hyperparams()

        structs = self.model_structure

        if self.build_setting == 'quick':
            structs = parse_quick(structs)

        self.is_multi_input_ = structs[0]['type'].lower() == "multi-input"

        return structs


    ### BUILDING THE MODEL ###

    def _add_multiinput_block(self,layer_specs,ind):
        '''
        Adds a multi-input block to the model.

        Must be the first layer if added.

        Parameters
        ----------
        layer_specs : dict
            A dictionary containing the keys ``'branches'`` and ``'merge_layer'``.

            The associated value of ``'branches'`` should be a non-empty list or tuple of the form::

                [[{'type': ...}, ...],
                [{'type': ...}, ...],
                ...]
            
            indicating the different branches.

            The associated value of ``'merge_layer'`` should be a keras.layers.Layer object that 
            controls how to merge the outputs of the branches.
        '''
        branches = layer_specs.get('branches')
        validate_branches(branches,ind)

        inputs = []
        outputs = []
        
        for branch_ind, branch in enumerate(branches):
            inp = kl.Input(shape=(
                                  self.input_shape_
                                  if not isinstance(self.input_shape_,list)
                                  else self.input_shape_[branch_ind]
                                )
                            )
            inputs.append(inp)

            out = inp
            for sub_ind, struct in enumerate(branch):
                out = add_block(struct,f"{ind}.{branch_ind}.{sub_ind}",out)
            
            outputs.append(out)

        return inputs, layer_specs.get('merge_layer')(outputs)

    def _build_model(self,structs):
        '''
        Builds the keras model from the given model structure.

        Parameters
        ----------
        structs : list or tuple
            A parsed ``model_structure``.
        
        Returns
        -------
        model : keras.Model
            The fully built and compiled model.
        
        Raises
        ------
        ValueError
            If the multi-input block did not come first.

            If the multi-output block did not come last.
        '''
        self.is_multi_output_ = False

        if structs[0]['type'].lower() == 'multi-input':
            struct = structs[0]
            structs = structs[1:]

            self.is_multi_input_ = True
            inputs,x = self._add_multiinput_block(struct.get('specs',struct),0)
        else:
            input_shape = self.input_shape_
            inputs = kl.Input(shape=input_shape)

            x = inputs

        for ind,struct in enumerate(structs):
            if struct['type'].lower() == 'multi-input':
                raise ValueError("Multi-input block must come first")

            if ind == len(structs) - 1:
                outputs = add_block(struct,ind,x)
            else:
                if struct['type'].lower() == 'multi-output':
                    raise ValueError("Multi-output block must come last")
                
                x = add_block(struct,ind,x)
        
        self.is_multi_output_ = isinstance(outputs,list)

        if self.is_multi_output_:
            self.output_shape_ = [
                keras.backend.int_shape(output)[1:]
                for output in outputs
            ]
        else:
            self.output_shape_ = keras.backend.int_shape(outputs)[1:]

        # Creates and compiles the model
        model = keras.Model(inputs, outputs)

        return model


    ### SKLEARN METHODS ###

    def fit(self, X, y, **fit_params):
        '''
        Trains the model on the given features and labels.

        Parameters
        ----------
        X : array-like
            The features of shape ``(n_samples, *input_shape_)`` 
            for single input or a list of features for multi-input

        y : array-like or list
            For non classification:

            The labels of shape ``(n_samples, *output_shape_)`` or
            ``(n_samples,)`` for single output, or a list of labels for
            multi-output.

            For classification:

            The class labels of shape ``(n_samples,)`` or a list of 
            class labels for multi-output.

        **fit_params
            Any additional fit parameters used in Keras.

        Returns
        -------
        self
            The trained estimator.
        
        Raises
        ------
        ValueError
            If X is sparse.
        '''
        if issparse(X):
            raise ValueError("Sparse input is not supported")

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        structs = self._prepare_structure()

        if self.is_multi_input_:
            X = [np.asarray(x) for x in X]
            expec_inp = [x.shape[1:] for x in X]
        else:
            X = np.asarray(X)
            expec_inp = X.shape[1:]

        self.input_shape_ = self.input_shape if self.input_shape is not None else expec_inp
        
        self.model_ = self._build_model(structs)
        self.model_.compile(optimizer=self._make_optimizer(),
                            loss=self.loss,
                            metrics=self.metrics)

        X,y = self._format_data(X,y)

        # If y is of shape (n_samples,), we need it to be of shape (n_samples,1)
        if self.is_multi_output_:
            self.y_was_1d_ = [
                np.asarray(target).ndim == 1
                for target in y
            ]

            y = [
                np.asarray(target).reshape(-1,1)
                if np.asarray(target).ndim == 1
                else np.asarray(target)
                for target in y
            ]
        else:
            y = np.asarray(y)

            self.y_was_1d_ = y.ndim == 1

            if y.ndim == 1:
                y = y.reshape(-1, 1)
        
        X,y = self._validate_data(X,y)

        callbacks = self._get_callbacks()

        if self.is_multi_output_:
            y = [target.astype(np.float32) for target in y]
        else:
            y = y.astype(np.float32)

        history = self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=self.shuffle,
            **fit_params,
        )

        # Stores the history, loss curve, and validation scores
        self.history_ = history.history
        self.loss_curve_ = history.history.get("loss")
        self.validation_scores_ = history.history.get("val_loss")

        return self

    def predict(self, X):
        '''
        Predicts the labels given the features.

        Parameters
        ----------
        X : array-like
            The features of shape ``(n_samples, *input_shape_)`` for single input.

            A list of features of shape ``(n_samples, *input_shape_[i])`` for multi-input.

        Returns
        -------
        y or [y1,...] : numpy.ndarray or list
            The labels of shape ``(n_samples, *output_shape_)`` or
            ``(n_samples,)`` for single output.

            For multi-output, it is a list of ndarrays with shape
            ``(n_samples, *output_shape_[i])`` or ``(n_samples,)``.
        
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
        self._check_is_fitted()
        X = self._format_data(X)
        X = self._validate_data(X)

        pred = self.model_.predict(X, verbose=0)

        if not self.is_multi_output_:
            if self.y_was_1d_:
                return pred.ravel()
        else:
            return [
                p.ravel() if was_1d else p
                for p, was_1d in zip(pred, self.y_was_1d_)
            ]

        return pred

    def score(self,X,y):
        '''
        Scores the model based on how it performs on given data.

        - For DeepEstimator, this returns the neg mse score.
        - For DeepRegressor, this returns the r2 score.
        - For DeepClassifier, this returns the accuracy score.

        Parameters
        ----------
        X : array-like
            The features of shape ``(n_samples, *input_shape_)`` for single input.

            A list of features of shape ``(n_samples, *input_shape_[i])`` for multi-input.

        y : array-like or list
            For non classification:
        
            The labels of shape ``(n_samples, *output_shape_)`` or
            ``(n_samples,)`` for single output, or a list of labels for
            multi-output.

            For classification:

            The class labels of shape ``(n_samples,)`` or a list of 
            labels for multi-output

        Returns
        -------
        score : float or None
            The score or weighted mean of scores (for multi-output).
        '''
        return compute_score(y,self.predict(X),
                             scoring_func=self.scoring_func,
                             weights=self.scoring_weights,
                             must_be_vector=self.must_be_vector)