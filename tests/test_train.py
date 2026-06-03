import numpy as np
import pandas as pd

import tif.train
import tif.utils


def test_pad_token_sequences_uses_utils_max_tokens() -> None:
    sequences = pd.Series([[1, 2, 3], [4] * (tif.utils.MAX_TOKENS + 5), None])

    matrix = tif.train._pad_token_sequences(sequences)

    assert matrix.shape == (3, tif.utils.MAX_TOKENS)
    assert matrix[0, :3].tolist() == [1, 2, 3]
    assert matrix[1, -1] == 4
    assert np.count_nonzero(matrix[2]) == 0


def test_sequence_plan_uses_lag_features() -> None:
    variables, steps = tif.train._sequence_plan(["a_lag_1", "a_lag_2", "b_lag_1", "plain"])

    assert variables == ["a"]
    assert steps == list(tif.train.SEQUENCE_STEPS)


def test_training_history_frame_extracts_neural_histories() -> None:
    summary = {
        "models": {
            "last_value": {"type": "baseline"},
            "numeric_mlp": {
                "type": "deep_numeric",
                "history": [
                    {"epoch": 1, "train_loss": 2.0, "validation_loss": 3.0},
                    {"epoch": 2, "train_loss": 1.0, "validation_loss": 2.0},
                ],
            },
        }
    }

    history = tif.train._training_history_frame(summary)

    assert history["model_name"].tolist() == ["numeric_mlp", "numeric_mlp"]
    assert history["validation_loss"].tolist() == [3.0, 2.0]
