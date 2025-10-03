"""Utility functions and callbacks shared across multiple analysis notebooks."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ttest_ind, ttest_rel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

from helper_functions.nn import create_nn


def binary_expected_calibration_error(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_bins: int = 10
) -> tf.Tensor:
    """Compute binary expected calibration error (ECE).

    Args:
        y_true: Tensor of true binary labels with shape (batch_size, 1).
        y_pred: Tensor of predicted probabilities with shape (batch_size, 1).
        num_bins: Number of probability bins to aggregate calibration error.

    Returns:
        Scalar tensor representing the ECE.
    """
    bin_edges = tf.linspace(0.0, 1.0, num_bins + 1)
    y_pred_flat = tf.reshape(y_pred, [-1])
    bin_indices = tf.searchsorted(bin_edges, y_pred_flat) - 1
    bin_indices = tf.clip_by_value(bin_indices, 0, num_bins - 1)

    mean_predictions = tf.math.unsorted_segment_mean(
        y_true, bin_indices, num_segments=num_bins
    )
    gathered_means = tf.gather(mean_predictions, bin_indices)
    abs_diff = tf.abs(y_pred_flat - tf.reshape(gathered_means, [-1]))
    return tf.reduce_mean(abs_diff)


class BinaryExpectedCalibrationError(tf.keras.losses.Loss):
    """Keras-compatible loss wrapper for the binary ECE."""

    def __init__(self, num_bins: int = 10, name: str = "binary_expected_calibration_error"):
        super().__init__(name=name)
        self.num_bins = num_bins

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return binary_expected_calibration_error(y_true, y_pred, num_bins=self.num_bins)


class MetricsCallback(tf.keras.callbacks.Callback):
    """Collect classification metrics at the end of each epoch on a hold-out set."""

    def __init__(self, X_test_num, X_test_encoded, Y_test):
        super().__init__()
        self.X_test_num = X_test_num
        self.X_test_encoded = X_test_encoded
        self.Y_test = Y_test
        self.sensitivity: List[float] = []
        self.specificity: List[float] = []
        self.auc: List[float] = []
        self.accuracy: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Mapping[str, float] | None = None) -> None:
        y_pred = self.model.predict([self.X_test_num, self.X_test_encoded], verbose=0)
        y_pred_binary = np.round(y_pred)
        tn, fp, fn, tp = confusion_matrix(self.Y_test, y_pred_binary).ravel()
        sensitivity = tp / (tp + fn + tf.keras.backend.epsilon())
        specificity = tn / (tn + fp + tf.keras.backend.epsilon())
        auc = roc_auc_score(self.Y_test, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        self.sensitivity.append(sensitivity)
        self.specificity.append(specificity)
        self.auc.append(float(auc))
        self.accuracy.append(accuracy)

        print(
            "Evaluation on Test Set:\n"
            f" - Sensitivity: {sensitivity:.4f}"
            f" - Specificity: {specificity:.4f}"
            f" - AUC: {auc:.4f}"
            f" - Accuracy: {accuracy:.4f}"
        )


def plot_history(history) -> None:
    """Plot training vs validation accuracy from a Keras History object."""
    training_accuracy = history.history.get("accuracy", [])
    validation_accuracy = history.history.get("val_accuracy", [])

    if not training_accuracy or not validation_accuracy:
        return

    epochs = range(1, len(training_accuracy) + 1)

    plt.figure(figsize=(12, 4))
    plt.plot(epochs, training_accuracy, "b", label="Training accuracy")
    plt.plot(epochs, validation_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def encode_and_split_new(X_tr, X_te, min_df: int = 10):
    """Split arrays with text column first into numeric and TF-IDF encoded text blocks."""
    X_train_text = X_tr[:, 0]
    X_test_text = X_te[:, 0]
    X_train_num = X_tr[:, 1:].astype("float32")
    X_test_num = X_te[:, 1:].astype("float32")

    vectorizer = TfidfVectorizer(min_df=min_df)
    X_train_encoded = vectorizer.fit_transform(X_train_text).toarray().astype("float32")
    X_test_encoded = vectorizer.transform(X_test_text).toarray().astype("float32")

    return X_train_num, X_test_num, X_train_encoded, X_test_encoded


def balance_classes(X_num, X_enc, Y):
    """Apply random over- and under-sampling to balance class distributions."""
    data = np.concatenate((X_enc, X_num, Y.reshape(-1, 1)), axis=1)
    classes = data[:, -6:-1]

    over_sampler = RandomOverSampler(sampling_strategy="auto")
    X_over, y_over = over_sampler.fit_resample(data, classes)

    under_sampler = RandomUnderSampler(sampling_strategy="auto")
    X_balanced, _ = under_sampler.fit_resample(X_over, y_over)

    X_enc_balanced = X_balanced[:, : X_enc.shape[1]]
    X_num_balanced = X_balanced[:, X_enc.shape[1] : X_enc.shape[1] + X_num.shape[1]]
    Y_balanced = X_balanced[:, -1]

    return X_num_balanced, X_enc_balanced, Y_balanced


def generate_one_hot(proportions: Sequence[float], N: int, seed: int | None = None) -> np.ndarray:
    """Sample one-hot encodings that match the requested proportions."""
    rng = np.random.default_rng(seed)
    M = len(proportions)
    one_hot_matrix = np.zeros((N, M), dtype=int)

    num_samples = (np.array(proportions) * N).astype(int)
    num_samples[-1] = N - num_samples[:-1].sum()

    indices = np.arange(N)
    rng.shuffle(indices)

    start = 0
    for col, num in enumerate(num_samples):
        end = start + num
        one_hot_matrix[indices[start:end], col] = 1
        start = end

    return one_hot_matrix


def verify_proportions(one_hot_matrix: np.ndarray, proportions: Sequence[float], categories: Sequence[str] | None = None) -> None:
    """Print actual versus expected category proportions."""
    N, _ = one_hot_matrix.shape
    counts = one_hot_matrix.sum(axis=0)
    actual_proportions = counts / N

    for i, (actual, expected) in enumerate(zip(actual_proportions, proportions)):
        label = categories[i] if categories is not None else f"Category {i}"
        print(f"{label}: Expected Proportion = {expected:.2f}, Actual Proportion = {actual:.2f}")


def ensure_unique_filename(save_path: str) -> str:
    """Append a counter to `save_path` if a file already exists."""
    base_name, ext = os.path.splitext(save_path)
    counter = 1
    candidate = save_path
    while os.path.exists(candidate):
        print(f"\nFile {candidate} already exists")
        candidate = f"{base_name}_{counter}{ext}"
        counter += 1

    if candidate != save_path:
        print(f"\nChanged file name to {candidate}")
    return candidate


def train_simultaneously(
    X_train_num,
    X_train_encoded,
    Y_train,
    X_test_num,
    X_test_encoded,
    Y_test,
    model1,
    model2,
    optimizer1,
    optimizer2,
    loss_fn,
    epochs: int = 4,
    batch_size: int = 32,
) -> None:
    """Custom training loop for paired models with and without race features."""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_num, X_train_encoded), Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(Y_train)).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(((X_test_num, X_test_encoded), Y_test))
        test_dataset = test_dataset.shuffle(buffer_size=len(Y_test)).batch(batch_size)

        train_acc_metric1 = tf.keras.metrics.BinaryAccuracy()
        test_acc_metric1 = tf.keras.metrics.BinaryAccuracy()
        train_loss_metric1 = tf.keras.metrics.Mean()
        test_loss_metric1 = tf.keras.metrics.Mean()

        train_acc_metric2 = tf.keras.metrics.BinaryAccuracy()
        test_acc_metric2 = tf.keras.metrics.BinaryAccuracy()
        train_loss_metric2 = tf.keras.metrics.Mean()
        test_loss_metric2 = tf.keras.metrics.Mean()

        for x_batch_train, y_batch_train in tqdm(train_dataset):
            x_train_num_batch, x_train_encoded_batch = x_batch_train
            x_train_num_subset = x_train_num_batch[:, :-5]

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                logits1 = model1([x_train_num_subset, x_train_encoded_batch], training=True)
                logits2 = model2([x_train_num_batch, x_train_encoded_batch], training=True)

                loss1 = loss_fn(y_batch_train, logits1)
                loss2 = loss_fn(y_batch_train, logits2)

            grads1 = tape1.gradient(loss1, model1.trainable_weights)
            optimizer1.apply_gradients(zip(grads1, model1.trainable_weights))

            grads2 = tape2.gradient(loss2, model2.trainable_weights)
            optimizer2.apply_gradients(zip(grads2, model2.trainable_weights))

            train_loss_metric1.update_state(loss1)
            train_loss_metric2.update_state(loss2)
            train_acc_metric1.update_state(y_batch_train, logits1)
            train_acc_metric2.update_state(y_batch_train, logits2)

        print(
            f"Epoch {epoch + 1}:\n"
            f"- Training Loss model 1 (no race): {train_loss_metric1.result().numpy().round(3)}\n"
            f"- Training Loss model 2 (race included): {train_loss_metric2.result().numpy().round(3)}\n"
            f"- Training Accuracy model 1: {train_acc_metric1.result().numpy() * 100:.2f}%\n"
            f"- Training Accuracy model 2: {train_acc_metric2.result().numpy() * 100:.2f}%"
        )

        for x_batch_test, y_batch_test in test_dataset:
            x_test_num_batch, x_test_encoded_batch = x_batch_test
            x_test_num_subset = x_test_num_batch[:, :-5]

            val_logits1 = model1([x_test_num_subset, x_test_encoded_batch], training=False)
            val_logits2 = model2([x_test_num_batch, x_test_encoded_batch], training=False)

            test_loss1 = loss_fn(y_batch_test, val_logits1)
            test_loss2 = loss_fn(y_batch_test, val_logits2)

            test_loss_metric1.update_state(test_loss1)
            test_loss_metric2.update_state(test_loss2)
            test_acc_metric1.update_state(y_batch_test, val_logits1)
            test_acc_metric2.update_state(y_batch_test, val_logits2)

        print(
            f"- Test Loss model 1 (no race): {test_loss_metric1.result().numpy().round(3)}\n"
            f"- Test Loss model 2 (race included): {test_loss_metric2.result().numpy().round(3)}\n"
            f"- Test Accuracy model 1: {test_acc_metric1.result().numpy() * 100:.2f}%\n"
            f"- Test Accuracy model 2: {test_acc_metric2.result().numpy() * 100:.2f}%"
        )

        for metric in (
            train_loss_metric1,
            train_loss_metric2,
            test_loss_metric1,
            test_loss_metric2,
            train_acc_metric1,
            train_acc_metric2,
            test_acc_metric1,
            test_acc_metric2,
        ):
            metric.reset_state()


def check_model(X_train, X_test, Y_train, Y_test, seed: int = 42):
    """Instantiate a baseline network after splitting inputs."""
    X_train_num, X_test_num, X_train_encoded, X_test_encoded = encode_and_split_new(X_train, X_test)
    model = create_nn([64, 32], [32], X_train_encoded.shape[1], X_train_num.shape[1], seed=seed)
    return model


def get_model_weights(model) -> List[np.ndarray]:
    """Flatten model weights into a list for comparison."""
    weights = []
    for layer in model.layers:
        for arr in layer.get_weights():
            weights.append(arr)
    return weights


def compare_weights(w1: Sequence[np.ndarray], w2: Sequence[np.ndarray]) -> None:
    """Print layer-by-layer equality diagnostics between two weight lists."""
    for idx, (weights1, weights2) in enumerate(zip(w1, w2)):
        print("Layer", idx)
        print(f"Shapes: {weights1.shape} , {weights2.shape}")
        if weights1.shape != weights2.shape:
            smaller_0 = min(weights1.shape[0], weights2.shape[0])
            smaller_1 = min(weights1.shape[1], weights2.shape[1])
            w1_slice = weights1[:smaller_0, :smaller_1]
            w2_slice = weights2[:smaller_0, :smaller_1]
            if not np.array_equal(w1_slice, w2_slice):
                print("- NOT EQUAL")
                print(w1_slice)
                print(w2_slice)
        else:
            if not np.array_equal(weights1, weights2):
                print("- NOT EQUAL")
                print(weights1)
                print(weights2)


def equate_weights(model_without_race, model_with_race, verbose: bool = True):
    """Copy shared weights from the no-race model into the race-aware model."""
    for layer_index, layer in enumerate(model_without_race.layers):
        if verbose:
            print(f"Layer {layer_index}:")
        for weight_index, weights_1 in enumerate(layer.get_weights()):
            weights_2 = model_with_race.layers[layer_index].get_weights()[weight_index]
            if verbose:
                print(f"Shapes {weight_index}: {weights_1.shape} , {weights_2.shape}")
            if not np.array_equal(weights_1, weights_2):
                weights_2[: weights_1.shape[0], : weights_1.shape[1]] = weights_1
                layer_weights = model_with_race.layers[layer_index].get_weights()
                layer_weights[weight_index] = weights_2
                model_with_race.layers[layer_index].set_weights(layer_weights)
                if verbose:
                    print("-----WEIGHTS UPDATED-----")
        if verbose:
            print()
    return model_without_race, model_with_race


def two_sample_t_test(arr1: np.ndarray, arr2: np.ndarray, alternative: str = "greater") -> float:
    """Welch's t-test between two independent samples."""
    _, p_value = ttest_ind(arr1, arr2, alternative=alternative, equal_var=False)
    return float(p_value)


def paired_t_test(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Paired t-test between two related samples."""
    _, p_value = ttest_rel(arr1, arr2, alternative="two-sided")
    return float(p_value)


def retraining_model_experiment(
    X_train_num,
    X_train_encoded,
    Y_train,
    X_test_num,
    X_test_encoded,
    Y_test,
    masks: Mapping[str, np.ndarray],
    races: Sequence[str],
    save_dir: str = "experiments_data/",
    save_name: str = "baseline",
    num_reruns: int = 50,
    p_values: Sequence[float] | None = None,
    save_increment: int = 5,
    epochs: int = 3,
    batch_size: int = 32,
    seed: int = 42,
):
    """Repeatedly retrain paired models and collect race-specific calibration metrics."""
    if p_values is None:
        p_values = (0.05, 0.01, 0.001, 0.0001)

    p_value_tables = {p: np.zeros((len(races), len(races))) for p in p_values}
    mean_differences = {race: np.zeros((num_reruns,)) for race in races}

    combined_probs_no_race = None
    combined_probs_with_race = None

    for i in range(num_reruns):
        print(f"\nRun {i + 1} of {num_reruns}\n")

        model1 = create_nn([64, 32], [32], X_train_encoded.shape[1], X_train_num.shape[1] - 5, seed=seed)
        model2 = create_nn([64, 32], [32], X_train_encoded.shape[1], X_train_num.shape[1], seed=seed)

        model1, model2 = equate_weights(model1, model2, verbose=False)

        optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = BinaryExpectedCalibrationError(num_bins=10)

        train_simultaneously(
            X_train_num,
            X_train_encoded,
            Y_train,
            X_test_num,
            X_test_encoded,
            Y_test,
            model1,
            model2,
            optimizer1,
            optimizer2,
            loss_fn,
            epochs=epochs,
            batch_size=batch_size,
        )

        probs_no_race = model1.predict([X_test_num[:, :-5], X_test_encoded])
        probs_with_race = model2.predict([X_test_num, X_test_encoded])

        if combined_probs_no_race is None:
            combined_probs_no_race = probs_no_race.reshape(1, -1)
            combined_probs_with_race = probs_with_race.reshape(1, -1)
        else:
            combined_probs_no_race = np.concatenate(
                (combined_probs_no_race, probs_no_race.reshape(1, -1)), axis=0
            )
            combined_probs_with_race = np.concatenate(
                (combined_probs_with_race, probs_with_race.reshape(1, -1)), axis=0
            )

        dict_races = {race: None for race in races}
        for race in races:
            differences = (probs_with_race[masks[race]] - probs_no_race[masks[race]]) * 100
            dict_races[race] = differences.flatten()
            mean_differences[race][i] = differences.mean()

        for p in p_values:
            for j, race_j in enumerate(races):
                for k, race_k in enumerate(races):
                    if j != k:
                        p_value_tables[p][j][k] += (
                            two_sample_t_test(dict_races[race_j], dict_races[race_k], alternative="less") < p
                        )
                    else:
                        p_value_tables[p][j][k] += (
                            paired_t_test(
                                probs_no_race[masks[race_j]],
                                probs_with_race[masks[race_k]],
                            )
                            < p
                        )

        if (i + 1) % save_increment == 0:
            print(f"Saving predictions for run {i + 1}")
            filename = f"{save_name}_predictions_{i + 1}.npz"
            modified_filename = ensure_unique_filename(filename)
            np.savez(os.path.join(save_dir, modified_filename),
                     probs_no_race=combined_probs_no_race,
                     probs_with_race=combined_probs_with_race)
            print(
                "--------------------------------------------------------------------\n\n"
                f"Predictions saved to {os.path.join(save_dir, modified_filename)}"
                "\n\n--------------------------------------------------------------------"
            )

    return mean_differences, p_value_tables


def plot_distribution(df: pd.DataFrame, column_name: str) -> None:
    """Plot a histogram for a numeric DataFrame column."""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return

    column_data = df[column_name]

    if not pd.api.types.is_numeric_dtype(column_data):
        print(f"Column '{column_name}' is not numeric and cannot be plotted.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(column_data.dropna(), color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.xlim(column_data.min(), column_data.max())
    plt.show()
