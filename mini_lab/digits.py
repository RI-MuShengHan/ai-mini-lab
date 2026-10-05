"""Reproduce the sklearn digits example, with a stratified held-out split."""
from functools import lru_cache
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def split_indices(labels, seed=42):
    indices = np.arange(len(labels))
    return train_test_split(indices, test_size=0.25, random_state=seed, stratify=labels)


@lru_cache(maxsize=1)
def train():
    data = load_digits()
    train_ids, test_ids = split_indices(data.target)
    # Fixed tutorial baseline: no searching over test scores.
    model = SVC(gamma=0.001)
    model.fit(data.data[train_ids], data.target[train_ids])
    predicted = model.predict(data.data[test_ids])
    dummy = DummyClassifier(strategy='most_frequent').fit(data.data[train_ids], data.target[train_ids])
    report = {
        'dataset': 'sklearn load_digits (8x8), not MNIST', 'seed': 42,
        'split': 'stratified 75% train / 25% test',
        'train_size': len(train_ids), 'test_size': len(test_ids),
        'accuracy': float(accuracy_score(data.target[test_ids], predicted)),
        'macro_f1': float(f1_score(data.target[test_ids], predicted, average='macro')),
        'dummy_accuracy': float(dummy.score(data.data[test_ids], data.target[test_ids])),
        'confusion_matrix': confusion_matrix(data.target[test_ids], predicted).tolist(),
        'parameters': {'model': 'SVC', 'gamma': 0.001},
        'note': 'Single fixed split; benchmark accuracy does not imply arbitrary handwriting accuracy.',
    }
    return model, data, test_ids, predicted, report


def sample(index=0):
    _, data, test_ids, predictions, report = train()
    if not 0 <= index < len(test_ids):
        raise ValueError('Sample index is outside the held-out test set.')
    idx = test_ids[index]
    return {'pixels': data.images[idx].tolist(), 'actual': int(data.target[idx]),
            'predicted': int(predictions[index]), 'index': index, 'count': len(test_ids),
            'accuracy': report['accuracy']}


def predict_pixels(pixels):
    arr = np.asarray(pixels, dtype=float)
    if arr.shape != (8, 8) or not np.isfinite(arr).all() or (arr < 0).any() or (arr > 16).any():
        raise ValueError('Expected an 8x8 array with finite pixel values from 0 to 16.')
    if not arr.any():
        raise ValueError('The drawing is empty.')
    return int(train()[0].predict(arr.reshape(1, -1))[0])
