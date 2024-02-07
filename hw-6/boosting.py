from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

np.random.seed(42) 

class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        indexes = np.random.randint(0, x.shape[0], int(self.subsample*x.shape[0]))
        x_bootstrap = x.copy()[indexes]
        y_bootstrap = y.copy()[indexes]
        s = -self.loss_derivative(y_bootstrap, predictions[indexes])

        new_model = self.base_model_class(**self.base_model_params)
        new_model.fit(x_bootstrap, s)
        new_model_predictions = new_model.predict(x)

        optimal_gamma = self.find_optimal_gamma(y, predictions, new_model_predictions)

        self.gammas.append(optimal_gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        counter_activation = False
        counter = 0

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

           
            if self.early_stopping_rounds is not None:
                if counter_activation:
                    if valid_loss < self.validation_loss.min():
                        counter = 0
                        self.validation_loss = np.full(self.early_stopping_rounds, np.inf)
                        counter_activation = False
                    else:
                        counter += 1
                        if counter >= self.early_stopping_rounds:
                            print('Early stopping!')
                            break
                else:
                    if valid_loss >= self.validation_loss.min():
                        counter_activation = True
                        counter = 1
                    self.validation_loss = np.roll(self.validation_loss, -1)
                    self.validation_loss[-1] = valid_loss


        if self.plot:
            plt.plot(self.history['train_loss'], label='train loss')
            plt.plot(self.history['valid_loss'], label='valid loss')
            plt.legend()
            plt.show

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])

        for gamma, model in zip(self.gammas, self.models):
            predictions += gamma * model.predict(x)
        
        return np.array([1-self.sigmoid(predictions), self.sigmoid(predictions)]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importance = list()
        for model in self.models:
            feature_importance.append(model.feature_importances_)

        feature_importance = np.mean(np.array(feature_importance), axis=0)

        return feature_importance

 
