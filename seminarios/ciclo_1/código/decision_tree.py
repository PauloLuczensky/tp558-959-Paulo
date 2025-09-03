import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from pso_optimizer.pso import PSOOptimizer
from grid_random_search_cv.hyperparameter_mappings_search_cv import (
    criterion_map_dt, splitter_map,
    min_samples_split_map_dt, max_features_map_dt
)

# ==============================
# Load dataset
# ==============================
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# 1) PSO Optimization
# ==============================
optimizer = PSOOptimizer(estimator="DT", random_state=42, random_seed=42)

best_hyperparameters, best_score, gbest_history, positions_history = optimizer.pso_hyperparameter_optimization(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    num_iterations=50,
    num_particles=100
)

print(f"\n[PSO] Best Score: {best_score}")
optimizer.get_hyperparameters(best_hyperparameters)
report_pso = optimizer.get_report(
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    best_hyperparameters=best_hyperparameters
)
print(f"[PSO] Classification Report:\n{report_pso}")

# ==============================
# 1b) PSO Animation
# ==============================
fig, ax = plt.subplots(figsize=(8,5))
scat = ax.scatter([], [], color='blue')
gbest_plot, = ax.plot([], [], 'r*', markersize=15)
ax.set_xlim(0, 2)   # índice do splitter
ax.set_ylim(5, 20)  # índice do max_depth (ajuste conforme o seu espaço de hiperparâmetros)
ax.set_xlabel("Splitter")
ax.set_ylabel("Max Depth")
ax.set_title("PSO Animation - Partículas convergindo")

def update(frame):
    # Ajuste dos índices para os hiperparâmetros que quer visualizar
    current_positions = np.array([[p[0], p[1]] for p in positions_history[frame]])
    scat.set_offsets(current_positions)
    
    fitness = [optimizer.evaluate_fitness(X_train, X_test, y_train, y_test, p) for p in positions_history[frame]]
    best_idx = np.argmax(fitness)
    gbest_plot.set_data(current_positions[best_idx,0], current_positions[best_idx,1])
    return scat, gbest_plot

ani = FuncAnimation(fig, update, frames=len(positions_history), interval=200, blit=True)
ani.save("pso_animation_dt.gif", writer=PillowWriter(fps=5))
plt.show()

# ==============================
# 2) Grid Search
# ==============================
dt_param_grid = {
    "criterion": list(criterion_map_dt.values()),
    "splitter": list(splitter_map.values()),
    "min_samples_split": list(min_samples_split_map_dt.values()),
    "max_features": list(max_features_map_dt.values())
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
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
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=dt_param_grid,
    n_iter=20,
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
