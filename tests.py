import os
import json
import app

def test_model_file_created():
    # Ensure the model file is created by the training script
    app.main()
    assert os.path.exists('models/model.pkl')

def test_model_score_regression():
    # 1. Train the model and get the new score
    score = app.main()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # 2. Load historical scores
    with open("model_scores.json", "r") as f:
        history = json.load(f)

    # 3. Compare against the last recorded score
    last_score = history[-1]["score"]
    assert score >= last_score, f"Model score {score} dropped below baseline {last_score}"
