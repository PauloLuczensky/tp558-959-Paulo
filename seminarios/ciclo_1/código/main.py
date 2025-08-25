from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from grid_random_search_cv.hyperparameter_mappings_search_cv import k_map
from pso_optimizer.pso import PSOOptimizer

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================
# 1) PSO Optimization
# ==============================
optimizer = PSOOptimizer(estimator="KNN", random_state=42, random_seed=42)

best_hyperparameters, best_score = optimizer.pso_hyperparameter_optimization(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    num_iterations=50,
    num_particles=100,
    c1=2.05,
    c2=2.05
)

print(f"\n[PSO] Best Score: {best_score}")
optimizer.get_hyperparameters(best_hyperparameters=best_hyperparameters)
report_pso = optimizer.get_report(
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    best_hyperparameters=best_hyperparameters
)
print(f"[PSO] Classification Report:\n{report_pso}")


# ==============================
# 2) Grid Search
# ==============================
param_grid = {
    "n_neighbors": list(k_map.keys()),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "l1", "l2"],
    "algorithm": ["brute", "kd_tree", "auto", "ball_tree"]
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"\n[GridSearch] Best Params: {grid_search.best_params_}")
print(f"[GridSearch] Best Score: {grid_search.best_score_}")
y_pred_grid = grid_search.best_estimator_.predict(X_test)
print(f"[GridSearch] Classification Report:\n{classification_report(y_test, y_pred_grid)}")


# ==============================
# 3) Randomized Search
# ==============================
param_dist = {
    "n_neighbors": list(k_map.keys()),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "l1", "l2"],
    "algorithm": ["brute", "kd_tree", "auto", "ball_tree"],
}

random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=param_dist,
    n_iter=20,          # número de amostragens
    cv=10,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"\n[RandomSearch] Best Params: {random_search.best_params_}")
print(f"[RandomSearch] Best Score: {random_search.best_score_}")
y_pred_random = random_search.best_estimator_.predict(X_test)
print(f"[RandomSearch] Classification Report:\n{classification_report(y_test, y_pred_random)}")
