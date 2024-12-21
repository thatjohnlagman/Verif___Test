import streamlit as st
from PIL import Image
import os
import helpers
import gdown
import base64

# Google Drive IDs - These should be the IDs of the FOLDERS on Google Drive
MODEL_FOLDER_ID = {
    "ai_text_detector": "1mqNt-jfusATtUZH8zUpu3nBKFXdIxyM0",
    "deepfake_audio_detection": "1utkXjbyiRlAamdWj3QrsANDxDNF4FgVh",
    "deepfake_image_detector": "1EWSUm5mmhavnX8GsM_7t8ZJ1xtXZA4mb",
    "phishing_detection": "1bw59K-0Xo1lmp_K-auRjLV_W-ESxEOhJ",
}

# Function to download model folders from Google Drive if they don't exist
def download_models(model_folder_ids):
    for model_name, folder_id in model_folder_ids.items():
        model_path = os.path.join("models", model_name)

        # Check if the model directory already exists
        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist

            # Download the folder from Google Drive
            gdown.download_folder(id=folder_id, output=model_path, quiet=False)

def init_streamlit():
    """Initialize Streamlit page configuration and styling"""
    st.set_page_config(
        page_title="VerifAI: Where AI Meets Authentication",
        page_icon=os.path.join("images", "Logo.png"),
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Custom styling
    st.markdown("""
        <style>
        /* Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

        *, *::before, *::after {
            font-family: 'Poppins', sans-serif;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            font-size: 24px;
            margin: 0;
            width: 100%;
        }

        /* Hide unnecessary elements */
        button[title="View fullscreen"] { visibility: hidden; }
        .reportview-container { margin-top: -2em; }
        #MainMenu { visibility: hidden; }
        .stDeployButton { display: none; }
        footer { visibility: hidden; }
        #stDecoration { display: none; }

        /* Navbar styling */
        .stApp {
            padding-top: 20px;
        }

        /* Progress bar color */
        .stProgress > div > div > div {
            background-color: #4284f2 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def display_navbar():
    """Display the navigation bar with 4 tabs"""

    # Inject custom CSS to force dark mode and style elements
    st.markdown(
        """
        <style>
            /* Force dark mode */
            html, body, [class*="ViewContainer"] {
                background-color: #0e1117; /* Streamlit's dark mode background color */
                color: #fafafa; /* Light text color for contrast */
            }

            /* Set the text color for the title in dark mode */
            .title-container h1 {
                color: rgb(255, 255, 255) !important; /* White text */
            }

            .image-container {
                text-align: center;
            }

            .image-container img {
                max-width: 80%; /* Adjust as needed */
            }

            /* Styling for the title and subtitle */
            .title-container {
                text-align: center;
            }

            .title-container h1 {
                font-size: 2.5em;
                margin-bottom: 0.1em;
            }

            .title-container h3 {
                font-weight: normal;
                color: gray;
                margin-top: -0.8em;
                margin-bottom: 0.7em;
            }

            /* Increase the font size of the tab titles */
            .stTabs [role="tab"] {
                font-size: 1.5em !important;
                font-weight: bold !important;
            }

            /* Set desired color for the navbar text */
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                color: #ff4500
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use the dark mode image
    image_path = os.path.join("images", "1.svg")  # Replace with your dark mode image
    text_color = "rgb(255, 255, 255)"  # White text for dark mode

    # Display the image at the top, centered with padding
    st.markdown(
        f"""
        <div class="image-container">
            <img src="data:image/svg+xml;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Center the title and subtitle using HTML and CSS
    st.markdown(
        f"""
        <div class="title-container">
            <h1 style="color: {text_color};">VerifAI: Where AI Meets Authentication</h1>
            <h3>Spot fakes and trust with confidence—powered by AI algorithms.</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navbar for navigation between models
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Deepfake Audio Detector",
            "Deepfake Image Detector",
            "AI Text Detector",
            "Phishing Link Detector",
        ]
    )

    return tab1, tab2, tab3, tab4


def phishing_detection_navbar(detector):
    st.title("Phishing Detection")
    st.write("Enter a URL and the model will classify it as **Benign** or **Dangerous**.")

    # Input for URL
    user_url = st.text_input("Enter URL:")

    # Button to trigger prediction
    if st.button("Classify URL"):
        if user_url:
            # Get prediction and confidence
            label, confidence = detector.check_link_validity(user_url)

            # Display the result with centered text and color-coded prediction
            color = "green" if label == "BENIGN" else "red"
            # Update "MALWARE" to "DANGEROUS"
            label = "DANGEROUS" if label == "MALWARE" else label

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="display: inline-block; margin-left: 20px;">Prediction: <span style="color: {color};">{label}</span></h3>
                    <p style="display: inline-block; font-size: 20px; margin-left: -6px;">Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True)
        else:
            st.error("Please enter a URL to classify.")

# Streamlit interface for the first menu
def deepfake_audio_detector_menu(detector):
    st.title("Audio Deepfake Detector")
    st.write("Upload an audio file, and the AI will classify it as **Real** or **Fake**.")

    # Upload audio
    uploaded_audio = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

    if uploaded_audio is not None:
        # Save the uploaded audio temporarily as .mp3
        temp_audio_path = os.path.join("temp_uploaded_audio.mp3")  # Save as .mp3
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_audio.read())

        # Display audio player
        st.audio(uploaded_audio, format="audio/mp3", start_time=0)

        # Make prediction
        try:
            predicted_label, confidence = detector.predict_audio_label(temp_audio_path)

            # Centered display of the result with colored label and confidence percentage
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="display: inline-block; margin-left: 20px;">Prediction: <span style="color: {'green' if predicted_label == 'REAL' else 'red'};">{predicted_label}</span></h3>
                    <p style="display: inline-block; font-size: 20px; margin-left: -6px;">Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing the audio file: {e}")

# Streamlit interface for the second menu
def deepfake_image_detector_menu(detector):
    st.title("Image Deepfake Detector")
    st.write("Upload an image, and the AI will classify it as **Real** or **Fake**.")

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Load the image
        image = Image.open(uploaded_image)

        # Center the image preview
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{image_to_base64(image)}" alt="Uploaded Image" width="400"/>
                <p>Uploaded Image</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Make prediction
        predicted_label, confidence = detector.predict(image)

        # Center the prediction and confidence output with your specific formatting
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h3 style="display: inline-block; margin-left: 20px;">Prediction: <span style="color: {'green' if predicted_label == 'REAL' else 'red'};">{predicted_label}</span></h3>
                <p style="display: inline-block; font-size: 20px; margin-left: -5px;">Confidence: {confidence*100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True)

# Function to convert the image to base64 for inline display in HTML
def image_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def ai_text_detector_menu(detector):
    st.title("AI Text Detector")
    st.write(
        "Upload or type a text, and the AI will classify it as **Human-generated** or **AI-generated**."
    )

    # Text input for manual typing or file upload
    input_choice = st.radio("Choose input method:", ("Type text", "Upload text file"))

    # Initialize user_text to ensure it always has a value
    user_text = ""

    if input_choice == "Type text":
        # Adjust the height parameter for the text area
        user_text = st.text_area("Enter text:", height=450)  # Adjust height (in pixels) here
    elif input_choice == "Upload text file":
        uploaded_file = st.file_uploader(
            "Upload a text file", type=["txt"], key="file_uploader_text"
        )
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode("utf-8")

    # Button to trigger prediction
    if st.button("Classify Text"):
        if user_text:
            # Make prediction
            prediction, human_prob, ai_prob = detector.classify_text(user_text)

            # Display the result with centered text and color-coded prediction
            if prediction == 0:
                predicted_label = "Human-generated"
                color = "green"
                confidence = human_prob
            else:
                predicted_label = "AI-generated"
                color = "red"
                confidence = ai_prob

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="display: inline-block; margin-left: 20px;">Prediction: <span style="color: {color};">{predicted_label}</span></h3>
                    <p style="display: inline-block; font-size: 20px; margin-left: -6px;">Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True)
        else:
            st.error("Please enter or upload some text before classifying.")

# Cache the model loading using the updated paths
@st.cache_resource
def load_phishing_detector():
    model_path = os.path.join("models", "phishing_detection")
    tokenizer_path = os.path.join("models", "phishing_detection")
    return helpers.PhishingDetector(model_path, tokenizer_path)

@st.cache_resource
def load_audio_detector():
    model_path = os.path.join("models", "deepfake_audio_detection", "model.h5")
    return helpers.DeepfakeAudioDetector(model_path=model_path)

@st.cache_resource
def load_image_detector():
    model_dir = os.path.join("models", "deepfake_image_detector")
    label_map = {0: "Real", 1: "Fake"}
    return helpers.DeepfakeImageDetector(model_dir=model_dir, label_map=label_map)

@st.cache_resource
def load_text_detector():
    model_path = os.path.join("models", "ai_text_detector")
    return helpers.AITextDetector(model_dir=model_path)

def main():
    # Initialize Streamlit app
    init_streamlit()

    # Download models if they don't exist locally
    download_models(MODEL_FOLDER_ID)

    # Display navbar and tabs
    tab1, tab2, tab3, tab4 = display_navbar()

    # Load the models using the caching functions (do this after downloading)
    audio_detector = load_audio_detector()
    image_detector = load_image_detector()
    text_detector = load_text_detector()
    phishing_detector = load_phishing_detector()

    with tab1:
        deepfake_audio_detector_menu(audio_detector)
    with tab2:
        deepfake_image_detector_menu(image_detector)
    with tab3:
        ai_text_detector_menu(text_detector)
    with tab4:
        phishing_detection_navbar(phishing_detector)

if __name__ == "__main__":
    main()