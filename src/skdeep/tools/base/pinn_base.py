import tensorflow as tf
from tensorflow.keras import models as km
from tensorflow.keras import metrics as kmet
from tensorflow.keras import ops as ko
import numpy as np

class PINN(km.Model):
    def __init__(self,
                 X_data,
                 mins,
                 maxs,
                 model,
                 calc_pde,
                 calc_bound_eqn,
                 constants,
                 data,
                 loss_weighting,
                 **kwargs):
        super().__init__(*kwargs)
        self.X_data = X_data
        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)

        self.model = model
        self.calc_pde = calc_pde
        self.calc_bound_eqn = calc_bound_eqn

        self.constants = constants
        self.data = data
        self.loss_weighting = loss_weighting

        self.loss_tracker = kmet.Mean()

    @property
    def metrics(self):
        return [self.loss_tracker]

    def get_loss(self,X_r):
        pde_loss = self.calc_pde(X_r)
        bc_loss = self.calc_bound_eqn(self.X_data)

        loss = 0

        for pde in pde_loss:
            loss += ko.sum(ko.square(pde))*self.loss_weighting['pde']

        for bc in bc_loss:
            loss += ko.sum(ko.square(bc))*self.loss_weighting['conditions']

        if self.data is not None:
            pred = self.model(self.data[:,0:-1])
            loss += ko.sum(ko.square(ko.reshape(self.data[:,-1],pred.shape) - pred))*self.loss_weighting['data']

        return loss

    def train_step(self,X_r):
        with tf.GradientTape() as tape:
            trainable_variables = self.trainable_variables + [val for val in self.constants.values() if val.trainable]
            loss = self.get_loss(X_r)

        grad_theta = tape.gradient(loss,trainable_variables)
        self.optimizer.apply_gradients(zip(grad_theta,trainable_variables))

        self.loss_tracker.update_state(loss)

        return {'loss':self.loss_tracker.result()}

    def test_step(self,X_r):
        loss = self.get_loss(X_r)
        self.loss_tracker.update_state(loss)

        return {'loss':self.loss_tracker.result()}

    def call(self,inputs):
        return self.model(inputs)

