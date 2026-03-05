from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.config import CFG

def make_model(model_name: str = "rf") -> Pipeline:
    """
    Returns an sklearn Pipeline with the chosen classifier.
    model_name: 'rf' or 'hgb'
    """
    model_name = model_name.lower().strip()

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=CFG.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])

    if model_name in {"hgb", "histgb", "histgradientboosting"}:
        clf = HistGradientBoostingClassifier(
            random_state=CFG.random_state,
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
        )
        return Pipeline([("clf", clf)])

    raise ValueError(f"Unknown model_name='{model_name}'. Use 'rf' or 'hgb'.")