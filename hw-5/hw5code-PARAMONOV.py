import numpy as np
from collections import Counter
# import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    act_sort = np.argsort(feature_vector)
    feature_vector = feature_vector[act_sort]
    target_vector = target_vector[act_sort]

    uniq_vec, indx = np.unique(feature_vector, return_index=True)

    if len(uniq_vec) <= 1:
        return [], [], -np.inf, -np.inf

    all_t = ((uniq_vec + np.roll(uniq_vec, -1))/2)[:-1]

    num_of_1s = np.cumsum(target_vector)
    s = np.arange(1, len(target_vector)+1)
    change_dot = indx[1:] - 1

    p_1 = (num_of_1s / s)[change_dot]
    h_1 = 1 - p_1**2 - (1 - p_1)**2
    parts = s[change_dot]/len(feature_vector)

    num_of_1s = np.cumsum(target_vector[::-1])
    change_dot_2 = len(feature_vector) - 2 - change_dot

    p_2 = (num_of_1s / s)[change_dot_2]
    h_2 = 1 - p_2**2 - (1 - p_2)**2

    gini = -parts * h_1 - (1-parts) * h_2
    
    ind = np.argmax(gini)

    return all_t, gini, all_t[ind], gini[ind]




class DecisionTree(BaseEstimator, TransformerMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self.current_depth = 0

    def _fit_node(self, sub_X, sub_y, node, cur_depth = 0):
        ### Ошибка №1: замена != на ==. Проверка на то, что все объекты относятся к одному классу
        if np.all(sub_y == sub_y[0]) or (self._max_depth is not None and cur_depth >= self._max_depth): ### <--- Реализация max_depth
            node["type"] = "terminal"
            node["class"] = sub_y[0]

            return 
        
        feature_best, threshold_best, gini_best, split = None, None, None, None


        ### Ошибка №2: индексация с 0
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():

                    ### Ошибка №3: в предыдущей версии могло возникнуть деление на 0, поменял местами шаги: если key not in clicks, то ratio[key] = 0
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_click / current_count
                    else:
                        ratio[key] = 0

                ### Ошибка №4: реализовал свой "трансформер"
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, np.arange(len(sorted_categories))))

                ### Ошибка №5: заменил np.array на np.fromiter, чтобы по feature_vector можно было итерироваться 
                feature_vector = np.fromiter(map(lambda x: categories_map[x], sub_X[:, feature]), dtype=int)

            else:
                raise ValueError


            ### Ошибка №6: если количество уникальных значений в признаке равно 1, то не делаем по нему разбиение
            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if ((gini_best is None or gini > gini_best) and self._min_samples_split is None) or (self._min_samples_split is not None and
                                                            sum(feature_vector < threshold) >= self._min_samples_split and 
                                                            sum(feature_vector >= threshold) >= self._min_samples_split
                                                            ): ### <--- Реализация min_samples_split
                
                feature_best = feature
                gini_best = gini

                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": ### Ошибка (№6): изначально было с маленькой буквы
                    threshold_best = list(map(lambda x: x[0], filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError
                                
        
        
        if feature_best is None:
            node["type"] = "terminal"

            ### Ошибка №7: в лист добавляется tuple, берем первый элемент 
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        ### Реализация _min_samples_leaf ###
        if (self._min_samples_leaf is not None and (sub_X[split].shape[0] <= self._min_samples_leaf or sub_X[np.logical_not(split)].shape[0] <= self._min_samples_leaf)):

            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], cur_depth + 1)

        ### Ошибка №8: передается не та выборка sub_y
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], cur_depth + 1)


    def _predict_node(self, x, node):

        if node['type'] == 'terminal':
            return node['class']
        
        elif 'categories_split' in node:

            spl = x[node['feature_split']] in node['categories_split']
            
        elif "threshold" in node:
            spl = x[node["feature_split"]] < node['threshold']

        if spl:
            return self._predict_node(x, node["left_child"])
            
        else:
            return self._predict_node(x, node["right_child"])
  

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))

        return np.array(predicted)
    
    def get_params(self, deep=False):
        return {'feature_types':self._feature_types}

