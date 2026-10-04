from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the 5 required classification model instances with default parameters.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform'
        )
    }

def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Returns meaningful, streamlined hyperparameter grids for tuning using GridSearchCV.
    """
    return {
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        },
        "Decision Tree": {
            "max_depth": [3, 5, 8],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [4, 6, 8],
            "min_samples_split": [2, 5]
        },
        "Support Vector Machine": {
            "C": [0.5, 1.0, 5.0],
            "kernel": ["linear", "rbf"]
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    }
