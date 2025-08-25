from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from pso_optimizer.pso import PSOOptimizer
from grid_random_search_cv.hyperparameter_mappings_search_cv import (
    criterion_map_dt,
    splitter_map,
    min_samples_split_map_dt,
    max_features_map_dt
)

# ---- Montando o param_grid a partir dos maps ----
dt_param_grid = {
    "criterion": list(criterion_map_dt.values()),
    "splitter": list(splitter_map.values()),
    "min_samples_split": list(min_samples_split_map_dt.values()),
    "max_features": list(max_features_map_dt.values())
}

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 1) PSO Optimization
# -------------------------------
optimizer = PSOOptimizer(estimator="DT", random_state=42, random_seed=42)
best_hyperparameters, best_score = optimizer.pso_hyperparameter_optimization(
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    num_iterations=50, num_particles=30, c1=2.05, c2=2.05
)

print(f"\n[PSO] Best Score: {best_score}")
print(f"[PSO] Best Hyperparameters: {optimizer.get_hyperparameters(best_hyperparameters)}")

# Report PSO
report_pso = optimizer.get_report(
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    best_hyperparameters=best_hyperparameters
)
print(f"\n[PSO] Classification Report:\n{report_pso}")


# -------------------------------
# 2) GridSearchCV
# -------------------------------
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=10,
    n_jobs=-1,
    scoring="accuracy"
)
grid_search.fit(X_train, y_train)

print(f"\n[GridSearch] Best Score: {grid_search.best_score_}")
print(f"[GridSearch] Best Params: {grid_search.best_params_}")

y_pred_grid = grid_search.best_estimator_.predict(X_test)
print(f"[GridSearch] Classification Report:\n{classification_report(y_test, y_pred_grid)}")


# -------------------------------
# 3) RandomizedSearchCV
# -------------------------------
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=dt_param_grid,
    n_iter=20,  # número de combinações aleatórias
    cv=10,
    n_jobs=-1,
    random_state=42,
    scoring="accuracy"
)
random_search.fit(X_train, y_train)

print(f"\n[RandomSearch] Best Score: {random_search.best_score_}")
print(f"[RandomSearch] Best Params: {random_search.best_params_}")

y_pred_random = random_search.best_estimator_.predict(X_test)
print(f"[RandomSearch] Classification Report:\n{classification_report(y_test, y_pred_random)}")
