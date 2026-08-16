import numpy as np

from defect_detection.feature_anomaly import (
    calculate_feature_distances,
    calculate_nearest_neighbor_distances,
    calculate_threshold,
    classify_anomalies,
)


def test_feature_distance():

    center = np.array([
        0.0,
        0.0,
    ])

    features = np.array([
        [0.0, 0.0],
        [3.0, 4.0],
    ])

    distances = calculate_feature_distances(
        features,
        center,
    )

    assert np.isclose(
        distances[0],
        0.0,
    )

    assert np.isclose(
        distances[1],
        5.0,
    )


def test_nearest_neighbor_distance():

    train_features = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])

    features = np.array([
        [3.0, 4.0],
    ])

    distances = (
        calculate_nearest_neighbor_distances(
            features,
            train_features,
        )
    )

    assert np.isclose(
        distances[0],
        5.0,
    )


def test_threshold():

    scores = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ])

    threshold = calculate_threshold(
        scores,
        percentile=80,
    )

    assert threshold > 4.0
    assert threshold <= 5.0


def test_classification():

    scores = np.array([
        1.0,
        3.0,
        5.0,
    ])

    predictions = classify_anomalies(
        scores,
        threshold=3.0,
    )

    expected = np.array([
        0,
        1,
        1,
    ])

    assert np.array_equal(
        predictions,
        expected,
    )