import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

# ==========================================================
# Developed By : Parth
# Roll No      : 241504
# Course       : BCA - Data Science (3rd Year)
# College      : Panipat Institute of Engineering & Technology
# Project      : Breast Cancer Prediction using Deep Learning
# ==========================================================

# Load Model & Scaler
model = tf.keras.models.load_model("breast_cancer_model.h5")
scaler = joblib.load("breast_cancer_scaler.pkl")

# Feature Names
feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "Radius Error",
    "Texture Error",
    "Perimeter Error",
    "Area Error",
    "Smoothness Error",
    "Compactness Error",
    "Concavity Error",
    "Concave Points Error",
    "Symmetry Error",
    "Fractal Dimension Error",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension",
]


# Prediction Function
def predict(*inputs):

    data = np.array(inputs).reshape(1, -1)

    data = scaler.transform(data)

    prediction = model.predict(data, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        result = "🔴 Malignant (Cancer Detected)"
        confidence = probability * 100
        color = "#ff4d4d"

    else:
        result = "🟢 Benign (No Cancer Detected)"
        confidence = (1 - probability) * 100
        color = "#00c853"

    html = f"""
    <div style="
        background:{color};
        padding:18px;
        border-radius:12px;
        color:white;
        text-align:center;
        font-size:22px;
        font-weight:bold;">
        {result}
        <br><br>
        Confidence : {confidence:.2f}%
    </div>
    """

    return html


# Custom Theme
css = """
body{
    background:#0f172a;
}

.gradio-container{
    max-width:1200px !important;
}

footer{
visibility:hidden;
}

h1{
text-align:center;
}

"""


inputs = []

for feature in feature_names:
    inputs.append(
        gr.Number(
            label=feature,
            value=0
        )
    )


with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🩺 Breast Cancer Prediction System

### Deep Learning Based Breast Cancer Detection

Predict whether a tumor is **Benign** or **Malignant** using a trained Neural Network.
""")

    with gr.Row():

        with gr.Column():
            gr.Markdown("## Enter Patient Details")
            input_components = inputs

        with gr.Column():
            gr.Markdown("## Prediction")
            output = gr.HTML()

            predict_btn = gr.Button(
                "Predict",
                variant="primary"
            )

            clear_btn = gr.ClearButton(
                input_components
            )

    predict_btn.click(
        fn=predict,
        inputs=input_components,
        outputs=output
    )

    gr.Markdown("""
---

### 👨‍💻 Developed By

**Parth**

**Roll No:** 241504

**Course:** BCA - Data Science (3rd Year)

**College:** Panipat Institute of Engineering & Technology (PIET), Samalkha

---

© 2026 Breast Cancer Prediction System
""")

if _name_ == "_main_":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
