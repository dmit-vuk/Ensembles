import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
import random


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        
        history = {'loss_train': [], 'loss_val': []}
        
        self.forest = []
        all_features = list(range(1, X.shape[1]))
        self.tree_features = []
        res_train = np.zeros(X.shape[0])
        if X_val is not None:
            res_val = np.zeros(X_val.shape[0])
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth = self.max_depth, **self.trees_parameters)
            
            boot_idx = np.random.randint(0, X.shape[0], X.shape[0])
            random.shuffle(all_features)
            
            tree.fit(X[boot_idx][:, all_features[:self.feature_subsample_size]], y[boot_idx])
            self.forest += [tree]
            self.tree_features.append(all_features[:self.feature_subsample_size])
            
            
            res_train += tree.predict(X[:, all_features[:self.feature_subsample_size]])
            history['loss_train'].append(self.rmse( y, res_train/(i+1) ))
            if X_val is not None:
                res_val += tree.predict(X_val[:, all_features[:self.feature_subsample_size]])
                history['loss_val'].append(self.rmse( y_val, res_val/(i+1) ))
        return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = np.zeros(X.shape[0])
        for i in range(len(self.forest)):
            tree = self.forest[i]
            features = self.tree_features[i]
            res += tree.predict(X[:, features])
        return res / self.n_estimators

    def rmse(self, y, y_pred):
        return np.sqrt(((y - y_pred)**2).mean())
    

class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        
        all_features = list(range(1, X.shape[1]))
        history = {'loss_train': [], 'loss_val': []}
        
        tree = DecisionTreeRegressor(max_depth = self.max_depth, **self.trees_parameters)

        random.shuffle(all_features)
        
        tree.fit(X[:, all_features[:self.feature_subsample_size]], y)
        res = tree.predict(X[:, all_features[:self.feature_subsample_size]])
        self.trees = [tree]
        self.tree_features = [all_features[:self.feature_subsample_size]]
        self.coef = [1]
        
        history['loss_train'].append(self.rmse(y, res))
        if X_val is not None:
            y_pred = self.predict(X_val)
            history['loss_val'].append(self.rmse(y_val, y_pred))
        
        for i in range(self.n_estimators - 1):
            tree = DecisionTreeRegressor(max_depth = self.max_depth, **self.trees_parameters)
            anti_gradient = y - res
            random.shuffle(all_features)
            
            tree.fit(X[:, all_features[:self.feature_subsample_size]], anti_gradient)
            y_pred = tree.predict(X[:, all_features[:self.feature_subsample_size]])
            self.coef += [minimize_scalar(lambda x: self.rmse(y, res + x*y_pred)).x]
            res += self.learning_rate * self.coef[i] * y_pred
            self.trees += [tree]
            self.tree_features.append(all_features[:self.feature_subsample_size])
            
            history['loss_train'].append(self.rmse(y, res))
            if X_val is not None:
                y_pred = self.predict(X_val)
                history['loss_val'].append(self.rmse(y_val, y_pred))
        return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        features = self.tree_features[0]
        res = self.trees[0].predict(X[:, features])
        for i in range(1, len(self.trees)):
            tree = self.trees[i]
            features = self.tree_features[i]
            res += self.learning_rate * self.coef[i] * tree.predict(X[:, features])
        return res
    
    def rmse(self, y, y_pred):
        return np.sqrt(((y - y_pred)**2).mean())