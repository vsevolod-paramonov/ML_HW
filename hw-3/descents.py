from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, delta: float = 1.0):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.delta = delta

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        if self.loss_function is LossFunction.MSE:
            return (2/x.shape[0]) * (x.T @ (x @ self.w - y))
        
        elif self.loss_function is LossFunction.LogCosh:
            return (1/x.shape[0]) * (np.tanh(x @ self.w - y) @ x)
        
        elif self.loss_function is LossFunction.MAE:

            grad = (x @ self.w - y)

            grad = ((grad > 0) * 1 + (grad == 0) * 0 + (grad < 0) * (-1))/x.shape[0] @ x

            return grad
        
        elif self.loss_function is LossFunction.Huber:

            d = self.delta

            grad = (y - x @ self.w )

            grad = -((abs(grad) < d) * (grad) + (abs(grad) >= d) * (d * np.sign(grad)))/x.shape[0] @ x

            return grad





    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:


        if self.loss_function is LossFunction.MSE:
        
            loss = np.mean((x @ self.w - y)**2)

        elif self.loss_function is LossFunction.LogCosh:

            loss =  np.mean(np.log(np.cosh(x @ self.w - y)))

        elif self.loss_function is LossFunction.MAE:

            loss = np.mean(abs(x @ self.w - y))
        
        elif self.loss_function is LossFunction.Huber:

            loss = (y - x @ self.w)

            loss = np.mean((abs(loss) < self.delta) * (0.5 * loss**2) + (abs(loss) >= self.delta) * (self.delta * abs(loss) - 0.5 * self.delta))

        return loss


    def predict(self, x: np.ndarray) -> np.ndarray:

        predictions = x @ self.w

        return predictions


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        
        learning_rate = self.lr()

        self.w -= learning_rate * gradient

        return -learning_rate*gradient

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        grad = super().calc_gradient(x, y)

        return grad


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE, delta: float = 1.0):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function, delta)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        # batch = self.batch_size

        # if self.batch_size > x.shape[0]:
        #     batch = x.shape[0]

        idx = np.random.randint(0, x.shape[0], self.batch_size)

        x = x[idx, :]
        y = y[idx]
        grad = super().calc_gradient(x, y)

        return grad



class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, delta: float = 1.0):
        super().__init__(dimension, lambda_, loss_function, delta)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)


    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        
        self.h = self.alpha * self.h + self.lr() * gradient

        self.w -= self.h

        return -self.h
       


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, delta: float = 1.0):
        super().__init__(dimension, lambda_, loss_function, delta)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:


        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)

        m_hat = self.m / (1 - (self.beta_1) ** self.iteration) 
        v_hat = self.v / (1 - (self.beta_2) ** self.iteration)


        step = (self.lr() * m_hat)/(np.sqrt(v_hat) + self.eps)
        
        self.w -= step

        return -step
    

class Nadam(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:


        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)

        m_hat = self.m / (1 - (self.beta_1) ** self.iteration) 
        v_hat = self.v / (1 - (self.beta_2) ** self.iteration)


        step = self.lr()/(np.sqrt(v_hat) + self.eps) * (self.beta_1 * m_hat + (1-self.beta_1)*gradient/(1 - self.beta_1 ** self.iteration))
        
        self.w -= step

        return -step
    
class AdaMax(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.u: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:


        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient

        m_hat = self.m / (1 - (self.beta_1) ** self.iteration) 

        self.u = (self.beta_2 * self.u > abs(gradient)) * (self.beta_2 * self.u) + (self.beta_2 * self.u <= abs(gradient)) * (abs(gradient))


        step = self.lr() * m_hat/(self.u + self.eps)
        
        self.w -= step

        return -step
    
class AMSGrad(VanillaGradientDescent):

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.v_hat: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:


        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient

        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)

        self.v_hat = (self.v_hat > self.v) * self.v_hat + (self.v_hat <= self.v) * self.v

        step = self.lr() * self.m / (np.sqrt(self.v_hat) + self.eps)

  
        self.w -= step

        return -step


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w

        l2_gradient[-1] = 0 ### СѓР±СЂР°С‚СЊ bias

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class NadamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AdaMaxReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AMSGradReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg, 
        'nadam': Nadam if not regularized else NadamReg,
        'adamax': AdaMax if not regularized else AdaMaxReg,
        'amsgrad': AMSGrad if not regularized else AMSGradReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
