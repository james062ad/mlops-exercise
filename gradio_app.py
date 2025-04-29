import gradio as gr
import app

def train_and_score():
    """
    Runs the full training pipeline (data gen → train → score)
    and returns the latest accuracy.
    """
    score = app.main()
    return f"Model accuracy: {score:.2f}"

# Build the Gradio interface
iface = gr.Interface(
    fn=train_and_score,
    inputs=[],
    outputs="text",
    title="Fraud Detector CI Demo",
    description="Train the model end-to-end and report accuracy."
)

if __name__ == "__main__":
    iface.launch()
