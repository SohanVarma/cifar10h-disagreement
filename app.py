import os
import json
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10H Human Disagreement Prediction",
    page_icon="🧠",
    layout="wide"
)

RESULTS_DIR = "results"
REPORT_PATH = "report.pdf"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEFAULT_METRICS = {
    "KL_mean": 0.260705,
    "KL_std": 0.664492,
    "JSD_mean": 0.051941,
    "JSD_std": 0.115524,
    "Cosine_mean": 0.933601,
    "Cosine_std": 0.196743,
    "Pearson_entropy": 0.416895,
    "Spearman_entropy": 0.459598,
    "Precision@100": 0.240,
    "Precision@200": 0.345,
    "Precision@500": 0.528,
}

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 900;
        color: #4DA3FF;
        line-height: 1.15;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 19px;
        color: #D6EAF8;
        margin-bottom: 1.2rem;
    }

    .section-card {
        background: linear-gradient(135deg, #102A43 0%, #1B4F72 100%);
        color: #FFFFFF !important;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #5DADE2;
        margin-bottom: 18px;
        font-size: 17px;
        line-height: 1.6;
    }

    .small-card {
        background-color: #102A43;
        color: #FFFFFF !important;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #5DADE2;
        box-shadow: 0 1px 8px rgba(0,0,0,0.25);
        min-height: 150px;
    }

    .small-card h4 {
        color: #85C1E9 !important;
        font-size: 21px;
        margin-bottom: 10px;
    }

    .small-card p,
    .small-card div {
        color: #FFFFFF !important;
    }

    .metric-card {
        background-color: #102A43;
        color: #FFFFFF !important;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #5DADE2;
        box-shadow: 0 1px 8px rgba(0,0,0,0.25);
    }

    .metric-value {
        font-size: 30px;
        font-weight: 900;
        color: #58D68D !important;
    }

    .metric-label {
        font-size: 14px;
        color: #EAF2F8 !important;
    }

    .tag {
        display: inline-block;
        background-color: #154360;
        color: #FFFFFF !important;
        padding: 6px 10px;
        margin: 4px 4px 4px 0;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        border: 1px solid #5DADE2;
    }

    .warning-box {
        background-color: #4A2C00;
        color: #FFFFFF !important;
        padding: 16px;
        border-radius: 12px;
        border-left: 6px solid #F39C12;
        margin-bottom: 14px;
    }

    .success-box {
        background-color: #0B3D2E;
        color: #FFFFFF !important;
        padding: 16px;
        border-radius: 12px;
        border-left: 6px solid #27AE60;
        margin-bottom: 14px;
    }

    div[data-testid="stTabs"] button {
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def load_image(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


def show_plot(title, filename, explanation, caption=None):
    st.subheader(title)
    img = load_image(filename)

    if img is not None:
        st.image(img, caption=caption if caption else title, use_container_width=True)
    else:
        st.warning(f"Missing file: results/{filename}")

    st.write(explanation)
    st.divider()


def load_metrics():
    metrics = DEFAULT_METRICS.copy()
    path = os.path.join(RESULTS_DIR, "test_metrics.json")

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
                metrics.update(loaded)
        except Exception:
            pass

    return metrics


metrics = load_metrics()

st.markdown(
    '<div class="main-title">Predicting Human Annotator Disagreement on CIFAR-10 Images</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A presentation-ready dashboard for modeling human uncertainty using CIFAR-10H soft labels.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-card">
    <b>Project Goal:</b> Build a deep learning model that predicts how humans disagree about an image label.
    Instead of outputting only one class, the model outputs a full probability distribution over all 10 CIFAR-10 classes.
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs([
    "🏠 Executive Summary",
    "🎯 Problem Setting",
    "📊 Dataset & EDA",
    "🧠 Model Architecture",
    "⚙️ Training Pipeline",
    "📈 Results",
    "🧪 Ablations",
    "🔬 Robustness",
    "🔥 Explainability",
    "🚨 Failure Analysis",
    "📄 Report"
])

with tabs[0]:
    st.header("Executive Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="small-card">
            <h4>What was built?</h4>
            <p>A CNN-based model that predicts a 10-class human label distribution for CIFAR-10 images.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="small-card">
            <h4>Why is it different?</h4>
            <p>It models disagreement and ambiguity instead of forcing every image into one hard class.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="small-card">
            <h4>Best model</h4>
            <p>ResNet-18 adapted for CIFAR-10, pretrained on CIFAR-10, fine-tuned on CIFAR-10H using a custom KL + entropy loss.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Key Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["KL_mean"]:.3f}</div>
                <div class="metric-label">KL Divergence ↓</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["JSD_mean"]:.3f}</div>
                <div class="metric-label">JSD ↓</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["Cosine_mean"]:.3f}</div>
                <div class="metric-label">Cosine Similarity ↑</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["Spearman_entropy"]:.3f}</div>
                <div class="metric-label">Spearman Entropy ↑</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("One-line Presentation Summary")
    st.success(
        "This project learns human uncertainty by predicting CIFAR-10H soft annotator distributions, producing richer and more interpretable outputs than a standard classifier."
    )

with tabs[1]:
    st.header("Problem Setting")

    st.markdown(
        """
        In standard CIFAR-10 classification, each image is treated as having one correct label.
        However, humans may disagree when an image is low-resolution, visually ambiguous, or resembles multiple categories.

        This project asks a different question:
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Standard Classifier")
        st.code("Question: What is this image?\nOutput: cat", language="text")
        st.write("A standard classifier produces one class or one confidence score.")

    with col2:
        st.subheader("Disagreement Predictor")
        st.code(
            "Question: What will people think this image is?\n\ncat: 0.60\ndog: 0.25\ndeer: 0.10\nhorse: 0.05",
            language="text"
        )
        st.write("This project predicts how human labels are distributed.")

    st.subheader("Mathematical Goal")
    st.latex(r"q(y|x) \approx p(y|x)")
    st.write(
        """
        Here, p(y|x) is the human annotator distribution from CIFAR-10H,
        and q(y|x) is the model-predicted distribution.
        """
    )

with tabs[2]:
    st.header("Dataset and Exploratory Data Analysis")

    st.markdown(
        """
        <span class="tag">CIFAR-10</span>
        <span class="tag">CIFAR-10H</span>
        <span class="tag">Soft labels</span>
        <span class="tag">Human disagreement</span>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Dataset Details")

    dataset_table = {
        "Dataset": ["CIFAR-10", "CIFAR-10H"],
        "Size": ["60,000 images", "10,000 images"],
        "Label Type": ["Hard labels", "Soft human label distributions"],
        "Use in Project": ["Pretraining", "Fine-tuning and evaluation"]
    }

    st.table(dataset_table)

    st.markdown(
        """
        <div class="warning-box">
        <b>Important:</b> CIFAR-10H aligns with the CIFAR-10 test set, not the training set.
        The project uses CIFAR-10 hard labels only for representation pretraining.
        </div>
        """,
        unsafe_allow_html=True
    )

    show_plot(
        "Entropy Distribution",
        "entropy_histogram.png",
        "Entropy measures the amount of disagreement among annotators. Most samples have low entropy, meaning people agree. The long tail represents ambiguous images where annotators disagree.",
        "Figure: Distribution of human label entropy in CIFAR-10H."
    )

    show_plot(
        "Per-Class Average Entropy",
        "per_class_entropy.png",
        "This plot reveals which classes tend to be more ambiguous. Animal classes such as cat, deer, bird, and dog tend to show higher disagreement because of visual similarity in 32×32 images.",
        "Figure: Average annotator disagreement per CIFAR-10 class."
    )

    show_plot(
        "Annotator Confusion Matrix",
        "annotator_confusion_matrix.png",
        "The confusion-style matrix summarizes the average human label distribution by class. It helps identify which categories humans are likely to confuse.",
        "Figure: Annotator distribution confusion matrix."
    )

    col1, col2 = st.columns(2)

    with col1:
        show_plot(
            "Low-Disagreement Images",
            "low_entropy_examples.png",
            "Low-entropy images are visually clear and most annotators agree on the same class."
        )

    with col2:
        show_plot(
            "High-Disagreement Images",
            "high_entropy_examples.png",
            "High-entropy images are ambiguous, blurry, or share features across multiple classes."
        )

with tabs[3]:
    st.header("Model Architecture")

    st.subheader("Backbone")
    st.write(
        """
        The main backbone is ResNet-18, modified for 32×32 CIFAR-10 images.
        A default ImageNet-style ResNet uses larger early downsampling, which can destroy information in small images.
        """
    )

    st.code(
        """
Input: 32×32 RGB image
    ↓
Modified ResNet-18
    - 3×3 first convolution
    - stride 1
    - no max pooling
    ↓
Feature vector
    ↓
MLP prediction head
    ↓
Softmax
    ↓
10-dimensional human label distribution
        """,
        language="text"
    )

    st.subheader("Prediction Head")
    st.write("Two prediction heads were considered:")

    st.markdown(
        """
        - **Linear head:** simple and interpretable, but lower capacity.
        - **MLP head:** higher capacity, better for modeling complex soft-label distributions.
        """
    )

    st.markdown(
        """
        <div class="success-box">
        <b>Final Choice:</b> ResNet-18 with MLP head because it gives more flexibility for predicting human disagreement patterns.
        </div>
        """,
        unsafe_allow_html=True
    )

with tabs[4]:
    st.header("Training Pipeline")

    st.subheader("Two-stage Training Strategy")

    st.code(
        """
Stage 1: CIFAR-10 hard-label pretraining
    Goal: Learn general visual features

Stage 2: CIFAR-10H soft-label fine-tuning
    Goal: Learn human disagreement distributions
        """,
        language="text"
    )

    st.subheader("Loss Functions")

    st.markdown("### 1. KL Divergence")
    st.latex(r"KL(p||q) = \sum_y p(y)\log\frac{p(y)}{q(y)}")
    st.write("Used as the primary distribution-matching baseline.")

    st.markdown("### 2. Jensen-Shannon Divergence")
    st.write("A symmetric and bounded version of KL divergence.")

    st.markdown("### 3. Custom Loss")
    st.latex(r"Loss = KL(p||q) + \lambda(H_{true} - H_{pred})^2")
    st.write(
        "The custom loss encourages the model to match both the full distribution and the amount of human disagreement."
    )

    show_plot(
        "Training and Validation Loss Curve",
        "loss_curve_custom_mlp.png",
        "The training loss decreases steadily. The validation loss stabilizes, showing that the model learns without completely overfitting.",
        "Figure: Training and validation loss over epochs."
    )

with tabs[5]:
    st.header("Evaluation Results")

    st.subheader("Metric Table")

    results_table = {
        "Metric": [
            "KL Divergence",
            "Jensen-Shannon Divergence",
            "Cosine Similarity",
            "Pearson Entropy Correlation",
            "Spearman Entropy Correlation",
            "Precision@100",
            "Precision@200",
            "Precision@500"
        ],
        "Value": [
            round(metrics["KL_mean"], 3),
            round(metrics["JSD_mean"], 3),
            round(metrics["Cosine_mean"], 3),
            round(metrics["Pearson_entropy"], 3),
            round(metrics["Spearman_entropy"], 3),
            round(metrics["Precision@100"], 3),
            round(metrics["Precision@200"], 3),
            round(metrics["Precision@500"], 3),
        ],
        "Interpretation": [
            "Lower is better; measures distribution mismatch.",
            "Lower is better; stable symmetric distribution metric.",
            "Higher is better; predicted distribution direction aligns well.",
            "Measures linear relation between true and predicted entropy.",
            "Measures rank relation for disagreement severity.",
            "Top 100 high-disagreement image retrieval.",
            "Top 200 high-disagreement image retrieval.",
            "Top 500 high-disagreement image retrieval.",
        ]
    }

    st.dataframe(results_table, use_container_width=True)

    show_plot(
        "Predicted vs True Entropy",
        "predicted_vs_true_entropy.png",
        "This scatter plot compares predicted disagreement with true human disagreement. The positive trend shows that the model can identify more ambiguous images.",
        "Figure: Predicted entropy versus true human entropy."
    )

    st.subheader("Result Interpretation")
    st.write(
        """
        The model achieves high cosine similarity, indicating strong alignment between predicted and true human distributions.
        KL and JSD are reasonably low, showing good distribution matching.
        Pearson and Spearman correlations show that the model captures uncertainty trends, although extremely ambiguous cases remain challenging.
        """
    )

with tabs[6]:
    st.header("Ablation Studies")

    st.write("Ablation studies help explain why the final design was selected.")

    ablation_table = {
        "Ablation": [
            "Backbone Initialization",
            "Loss Function",
            "Prediction Head"
        ],
        "Comparison": [
            "Random initialization vs CIFAR-10 pretraining",
            "KL vs JSD vs Custom KL + entropy loss",
            "Linear head vs MLP head"
        ],
        "Finding": [
            "CIFAR-10 pretraining improves convergence and stability.",
            "Custom loss better captures both distribution and uncertainty.",
            "MLP head models soft distributions better than a single linear layer."
        ]
    }

    st.table(ablation_table)

    st.subheader("Why this matters")
    st.write(
        "These ablations show that the final design is not arbitrary. Each component supports the main task of modeling human disagreement."
    )

with tabs[7]:
    st.header("Robustness Checks")

    st.write(
        """
        Robustness checks test whether model uncertainty behaves sensibly under degraded input conditions.
        If an image becomes noisy or unclear, predicted entropy should generally increase.
        """
    )

    show_plot(
        "Robustness to Gaussian Noise",
        "robustness_noise.png",
        "As Gaussian noise severity increases, predicted entropy increases. This is expected because noisy images are harder to interpret.",
        "Figure: Entropy response to noise corruption."
    )

    show_plot(
        "Robustness to Gaussian Blur",
        "robustness_blur.png",
        "Blur removes visual details. The model initially increases uncertainty strongly, showing sensitivity to loss of object detail.",
        "Figure: Entropy response to blur corruption."
    )

    show_plot(
        "Robustness to Contrast Reduction",
        "robustness_contrast.png",
        "Lower contrast makes object boundaries less clear. The model responds with increased predicted entropy at higher severities.",
        "Figure: Entropy response to contrast corruption."
    )

with tabs[8]:
    st.header("Grad-CAM Explainability")

    st.write(
        """
        Grad-CAM helps visualize which image regions influenced the model's decision.
        This is useful for checking whether the model focuses on meaningful object regions.
        """
    )

    gradcam_files = [
        "gradcam_example_0.png",
        "gradcam_example_1.png",
        "gradcam_example_2.png",
        "gradcam_example_3.png",
        "gradcam_example_4.png",
        "gradcam_example_5.png",
    ]

    cols = st.columns(2)

    for i, file in enumerate(gradcam_files):
        with cols[i % 2]:
            img = load_image(file)
            if img is not None:
                st.image(img, caption=file, use_container_width=True)
            else:
                st.warning(f"Missing file: results/{file}")

    st.subheader("Interpretation")
    st.write(
        """
        Clear images usually produce focused heatmaps, while ambiguous images produce more diffuse attention.
        This supports the idea that model uncertainty is connected to visual ambiguity.
        """
    )

with tabs[9]:
    st.header("Failure Case Analysis")

    st.write(
        """
        Failure cases are important because they show where the model still struggles.
        """
    )

    st.markdown(
        """
        Common failure reasons:
        - **Ambiguous object identity:** image resembles multiple classes.
        - **Low image quality:** CIFAR-10 images are only 32×32.
        - **Multi-object content:** more than one object may be visible.
        - **Boundary cases:** visually similar classes such as cat/dog or deer/horse.
        """
    )

    st.subheader("Main limitation")
    st.warning(
        "The model captures general disagreement trends, but very high-entropy images remain difficult because ambiguity is sometimes semantic, not just visual."
    )

with tabs[10]:
    st.header("Report and Submission")

    st.write("Download the final PDF report below.")

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "rb") as f:
            st.download_button(
                label="📄 Download Full Project Report",
                data=f,
                file_name="CIFAR10H_Human_Disagreement_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("report.pdf not found in the main project folder.")

    st.subheader("Recommended GitHub Submission Files")
    st.code(
        """
app.py
README.md
requirements.txt
report.pdf
src/
results/
        """,
        language="text"
    )

    st.success(
        "This dashboard is designed for presentation. It explains the project objective, dataset, methodology, results, robustness, explainability, and failure analysis."
    )