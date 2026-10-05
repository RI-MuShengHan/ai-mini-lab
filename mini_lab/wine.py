"""StandardScaler + logistic regression on the bundled UCI Wine dataset."""
from functools import lru_cache
import numpy as np
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline as sklearn_pipeline
from sklearn.preprocessing import StandardScaler


def split_indices(target):
    return train_test_split(np.arange(len(target)), test_size=0.25,
                            random_state=42, stratify=target)


def make_pipeline():
    return sklearn_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


@lru_cache(maxsize=1)
def train():
    data = load_wine()
    train_ids, test_ids = split_indices(data.target)
    model = make_pipeline()
    model.fit(data.data[train_ids], data.target[train_ids])
    predicted = model.predict(data.data[test_ids])
    dummy = DummyClassifier(strategy='most_frequent').fit(data.data[train_ids], data.target[train_ids])
    # CV uses only the training portion; each fold refits its own scaler.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(make_pipeline(), data.data[train_ids], data.target[train_ids],
                             cv=cv, scoring='f1_macro')
    report = {
        'dataset': 'UCI Wine via sklearn.load_wine',
        'split': 'stratified 75/25, seed=42; preprocessing fit on training only',
        'train_size': len(train_ids), 'test_size': len(test_ids), 'features': 13,
        'accuracy': float(accuracy_score(data.target[test_ids], predicted)),
        'macro_f1': float(f1_score(data.target[test_ids], predicted, average='macro')),
        'dummy_accuracy': float(dummy.score(data.data[test_ids], data.target[test_ids])),
        'train_cv_macro_f1': scores.tolist(), 'train_cv_mean': float(scores.mean()),
        'confusion_matrix': confusion_matrix(data.target[test_ids], predicted).tolist(),
        'parameters': {'C': 1.0, 'max_iter': 1000},
        'note': 'Small historical 3-class dataset; one fixed test split, no claim about production use.'
    }
    return model, data, train_ids, test_ids, report


def predict_features(values):
    try: x = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc: raise ValueError('Expected 13 numeric features.') from exc
    if x.shape != (13,) or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError('Expected 13 finite, positive numeric features.')
    model, data, _, _, _ = train()
    probabilities = model.predict_proba([x])[0]
    winner = int(probabilities.argmax())
    scaled = model['standardscaler'].transform([x])[0]
    contributions = scaled * model['logisticregression'].coef_[winner]
    order = np.argsort(np.abs(contributions))[::-1][:5]
    return {'predicted': str(data.target_names[winner]),
            'scores': {str(name): round(float(score), 4)
                       for name, score in zip(data.target_names, probabilities)},
            'features': [{'name': str(data.feature_names[i]),
                          'contribution': round(float(contributions[i]), 4)} for i in order]}


def sample(index=0):
    _, data, _, test_ids, report = train()
    if not 0 <= index < len(test_ids): raise ValueError('Sample index out of range.')
    sample_id = test_ids[index]
    values = data.data[sample_id].tolist()
    return {'index': index, 'count': len(test_ids), 'values': values,
            'names': list(data.feature_names), 'actual': str(data.target_names[data.target[sample_id]]),
            'accuracy': report['accuracy'], **predict_features(values)}
