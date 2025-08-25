from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from grid_random_search_cv.hyperparameter_mappings_search_cv import (
    k_map,
    distance_metric_map,
    weighting_method_map,
    algorithm_map,
    criterion_map_rf,
    min_samples_split_map_rf,
    max_features_map_rf,
    criterion_map_dt,
    splitter_map,
    min_samples_split_map_dt,
    max_features_map_dt,
    kernel_map,
    gamma_map,
    shrinking_map,
    decision_function_shape_map
)


class BaseSearchOptimizer:
    def __init__(self, estimator, random_state=42):
        self.estimator = estimator
        self.random_state = random_state
        self.best_estimator = None
        self.best_params = None
        self.best_score_ = None

    def get_hyperparameters(self):
        print("Optimal hyperparameters:")
        if self.estimator == "KNN":
            print(f"K: {k_map[self.best_params['n_neighbors']]}")
            print(f"Distance Metric: {distance_metric_map[self.best_params['distance_metric']]}")
            print(f"Weighting Method: {weighting_method_map[self.best_params['weights']]}")
            print(f"Algorithm: {algorithm_map[self.best_params['algorithm']]}")
            print(f"Leaf Size: {self.best_params['leaf_size']}")
            print(f"P Value: {self.best_params['p']}")
        elif self.estimator == "RF":
            print(f"n_estimators: {self.best_params['n_estimators']}")
            print(f"max_depth: {self.best_params['max_depth']}")
            print(f"Criterion: {criterion_map_rf[self.best_params['criterion']]}")
            print(f"min_samples_split: {min_samples_split_map_rf[self.best_params['min_samples_split']]}")
            print(f"min_samples_leaf: {self.best_params['min_samples_leaf']}")
            print(f"min_weight_fraction_leaf: {self.best_params['min_weight_fraction_leaf']}")
            print(f"max_features: {max_features_map_rf[self.best_params['max_features']]}")
        elif self.estimator == "DT":
            print(f"Splitter: {splitter_map[self.best_params['splitter']]}")
            print(f"max_depth: {self.best_params['max_depth']}")
            print(f"Criterion: {criterion_map_dt[self.best_params['criterion']]}")
            print(f"min_samples_split: {min_samples_split_map_dt[self.best_params['min_samples_split']]}")
            print(f"min_samples_leaf: {self.best_params['min_samples_leaf']}")
            print(f"min_weight_fraction_leaf: {self.best_params['min_weight_fraction_leaf']}")
            print(f"max_features: {max_features_map_dt[self.best_params['max_features']]}")
        elif self.estimator == "SVC":
            print(f"C: {self.best_params['C']}")
            print(f"Kernel: {kernel_map[self.best_params['kernel']]}")
            print(f"Degree: {self.best_params['degree']}")
            print(f"Gamma: {gamma_map[self.best_params['gamma']]}")
            print(f"Shrinking: {shrinking_map[self.best_params['shrinking']]}")
            print(f"Decision Function Shape: {decision_function_shape_map[self.best_params['decision_function_shape']]}")

    def get_report(self, X_train, X_test, y_train, y_test):
        y_pred = self.best_estimator.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    def _create_estimator(self, params):
        if self.estimator == "KNN":
            return KNeighborsClassifier(
                n_neighbors=k_map[params["n_neighbors"]],
                metric=distance_metric_map[params["distance_metric"]],
                weights=weighting_method_map[params["weights"]],
                algorithm=algorithm_map[params["algorithm"]],
                leaf_size=params["leaf_size"],
                p=params["p"]
            )
        elif self.estimator == "RF":
            return RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                criterion=criterion_map_rf[params["criterion"]],
                min_samples_split=min_samples_split_map_rf[params["min_samples_split"]],
                min_samples_leaf=params["min_samples_leaf"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                max_features=max_features_map_rf[params["max_features"]]
            )
        elif self.estimator == "DT":
            return DecisionTreeClassifier(
                random_state=self.random_state,
                splitter=splitter_map[params["splitter"]],
                max_depth=params["max_depth"],
                criterion=criterion_map_dt[params["criterion"]],
                min_samples_split=min_samples_split_map_dt[params["min_samples_split"]],
                min_samples_leaf=params["min_samples_leaf"],
                min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
                max_features=max_features_map_dt[params["max_features"]]
            )
        elif self.estimator == "SVC":
            return SVC(
                random_state=self.random_state,
                C=params["C"],
                kernel=kernel_map[params["kernel"]],
                degree=params["degree"],
                gamma=gamma_map[params["gamma"]],
                shrinking=shrinking_map[params["shrinking"]],
                decision_function_shape=decision_function_shape_map[params["decision_function_shape"]]
            )
        else:
            raise ValueError("Estimator not supported.")


class GridSearchOptimizer(BaseSearchOptimizer):
    def fit(self, X_train, y_train, param_grid, cv=5, scoring="accuracy"):
        model = self._create_estimator(param_grid)  # modelo dummy só para interface
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.best_estimator = self._create_estimator(grid_search.best_params_)
        self.best_estimator.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        return self.best_params, self.best_score_


class RandomSearchOptimizer(BaseSearchOptimizer):
    def fit(self, X_train, y_train, param_distributions, n_iter=50, cv=5, scoring="accuracy"):
        model = self._create_estimator(param_distributions)  # modelo dummy só para interface
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=n_iter,
            cv=cv, scoring=scoring, n_jobs=-1, random_state=self.random_state
        )
        random_search.fit(X_train, y_train)

        self.best_estimator = self._create_estimator(random_search.best_params_)
        self.best_estimator.fit(X_train, y_train)
        self.best_params = random_search.best_params_
        self.best_score_ = random_search.best_score_
        return self.best_params, self.best_score_
