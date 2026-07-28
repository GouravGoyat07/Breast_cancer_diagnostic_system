
import os
import gradio as gr
import numpy as np
import tensorflow as tf
import joblib

# =====================================================
# Developed By : Parth
# Roll No      : 241504
# Course       : BCA - Data Science (3rd Year)
# College      : Panipat Institute of Engineering & Technology, Samalkha
# Project      : Breast Cancer Prediction using Deep Learning
# =====================================================

model = tf.keras.models.load_model("breast_cancer_model.h5")
scaler = joblib.load("breast_cancer_scaler.pkl")

feature_names = [
    "Mean Radius","Mean Texture","Mean Perimeter","Mean Area",
    "Mean Smoothness","Mean Compactness","Mean Concavity",
    "Mean Concave Points","Mean Symmetry","Mean Fractal Dimension",
    "Radius Error","Texture Error","Perimeter Error","Area Error",
    "Smoothness Error","Compactness Error","Concavity Error",
    "Concave Points Error","Symmetry Error","Fractal Dimension Error",
    "Worst Radius","Worst Texture","Worst Perimeter","Worst Area",
    "Worst Smoothness","Worst Compactness","Worst Concavity",
    "Worst Concave Points","Worst Symmetry","Worst Fractal Dimension"
]

def predict(*values):
    try:
        data = np.array(values, dtype=float).reshape(1, -1)
        data = scaler.transform(data)
        prob = float(model.predict(data, verbose=0)[0][0])

        if prob >= 0.5:
            result = "🔴 Malignant (Cancer Detected)"
            confidence = prob * 100
            color = "#dc2626"
        else:
            result = "🟢 Benign (No Cancer Detected)"
            confidence = (1 - prob) * 100
            color = "#16a34a"

        return f"""
        <div style='background:{color};padding:25px;border-radius:15px;
        color:white;text-align:center;box-shadow:0 8px 20px rgba(0,0,0,.25);'>
            <h2>{result}</h2>
            <h3>Confidence</h3>
            <h1>{confidence:.2f}%</h1>
            <p><b>Educational Use Only</b></p>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red'><b>Error:</b> {e}</div>"

css = """
body{
background:linear-gradient(135deg,#0f172a,#1d4ed8);
}
.gradio-container{
max-width:1350px!important;
margin:auto;
}
footer{display:none!important;}
.title{
text-align:center;
font-size:40px;
font-weight:bold;
color:white;
}
.sub{
text-align:center;
color:#e5e7eb;
font-size:18px;
margin-bottom:20px;
}
.card{
background:black;
padding:18px;
border-radius:15px;
margin-top:10px;
box-shadow:0 4px 15px rgba(0,0,0,.15);
}
.dev{
background:#0f172a;
color:white;
padding:20px;
border-radius:15px;
text-align:center;
margin-top:20px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.HTML("""
    <div class='title'>🩺 Breast Cancer Prediction System</div>
    <div class='sub'>Deep Learning Based Diagnosis Assistant</div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("📋 Enter Patient Measurements", open=True):
                inputs = []
                for f in feature_names:
                    inputs.append(gr.Number(label=f, value=0))

        with gr.Column(scale=1):
            gr.HTML("""
            <div class='card'>
            <h3>ℹ️ Information</h3>
            <p><b>🟢 Benign:</b> Non-cancerous tumor.</p>
            <p><b>🔴 Malignant:</b> Cancerous tumor requiring medical evaluation.</p>
            <p><b>Note:</b> This application is for educational purposes only.</p>
            </div>
            """)
            output = gr.HTML()
            with gr.Row():
                predict_btn = gr.Button("🔍 Predict", variant="primary")
                gr.ClearButton(inputs, value="🗑 Clear")

    predict_btn.click(predict, inputs=inputs, outputs=output)

    gr.HTML("""
    <div class='dev'>
    <h2>👨‍💻 Developed By</h2>
    <h3>Parth</h3>
    <p><b>Roll No:</b> 241504</p>
    <p><b>Course:</b> BCA - Data Science (3rd Year)</p>
    <p><b>College:</b> Panipat Institute of Engineering & Technology, Samalkha</p>
    <hr>
    <p>© 2026 Breast Cancer Prediction System</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
