import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

import app  # your training script

def plot_score_history():
    # Load the historical scores
    df = pd.read_json("model_scores.json")
    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(df["version"], df["score"], marker="o")
    ax.set_title("Model Accuracy Over Versions")
    ax.set_xlabel("Version")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig

def run_pipeline():
    # Run the full training script and get latest score
    score = app.main()
    # Return both the textual result and the updated plot
    return f"Model accuracy: {score:.2f}", plot_score_history()

iface = gr.Interface(
    fn=run_pipeline,
    inputs=None,
    outputs=["text", "plot"],
    title="Fraud Detector CI Demo",
    description="Click **Generate** to train the model end-to-end and see both the latest accuracy and a chart of how it’s improved over time."
)

if __name__ == "__main__":
    iface.launch()
