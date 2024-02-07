from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent

import time


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:

        self.iterat = 0
        step = 1
        ev_norm = np.inf
        self.loss_history.append(self.descent.calc_loss(x,y))
        

        while True:

            if np.sum(np.isnan(step)) != 0:
                break

            if ev_norm <= self.tolerance:
                break

            if self.iterat >= self.max_iter:
                break

            step = self.descent.step(x, y)
            ev_norm = step.T @ step
            self.loss_history.append(self.descent.calc_loss(x,y))

            self.iterat += 1



    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
  
        return self.descent.calc_loss(x, y)
