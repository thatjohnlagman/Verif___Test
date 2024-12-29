import streamlit as st
from PIL import Image
import os
import helpers
import gdown
import nltk
nltk.download('stopwords')


# Google Drive IDs - These should be the IDs of the FOLDERS on Google Drive
MODEL_FOLDER_ID = {
    "ai_text_detector": "1N1EkWbTd8S3UiicvNM1eI8dWn21XPH2T",
    "deepfake_audio_detection": "1utkXjbyiRlAamdWj3QrsANDxDNF4FgVh",
    "deepfake_image_detector": "1EWSUm5mmhavnX8GsM_7t8ZJ1xtXZA4mb",
    "phishing_detection": "1Bhmcb6TPZlDKpBjS8xA4tdz_awtE2eup",
}

# Google Drive folder IDs for test files
IMAGE_TEST_FILES_FOLDER_ID = "10_ElyRhMRkV2sDXRt3JeBwLOadRXhsZY"
AUDIO_TEST_FILES_FOLDER_ID = "1X0Dl4o2Ecd5Aez3OPecs0ASeCeCzkQG-"


def download_models(model_folder_ids):
    for model_name, folder_id in model_folder_ids.items():
        model_path = os.path.join("models", model_name)

        # Check if the model directory already exists
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)  # Create the specific model directory if it doesn't exist
            print(f"Downloading {model_name} model from Google Drive...")
            gdown.download_folder(id=folder_id, output=model_path, quiet=False)

        # Verify that the expected model file exists for ai_text_detector
        if model_name == "ai_text_detector":
            expected_file = "ai_text_detector_model.pkl"  # File expected in this folder
            model_file = os.path.join(model_path, expected_file)
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Model file {model_file} not found after download. "
                    f"Ensure the file is available in the Google Drive folder: {folder_id}"
                )
            else:
                print(f"Verified: {model_file} is ready to use.")

        # Optional: Print contents of the downloaded directory for debugging
        print(f"Contents of {model_path}: {os.listdir(model_path)}")


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
    # Display the image at the top
    image_path = os.path.join("images", "1.svg")
    st.image(image_path, use_container_width=True)

    # Center the title and subtitle using HTML and CSS
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="font-size: 2.5em; margin-bottom: 0.1em;">VerifAI: Where AI Meets Authentication</h1>
            <h3 style="font-weight: normal; color: gray; margin-top: -0.8em; margin-bottom: 0.7em;">Spot fakes and trust with confidence—powered by AI algorithms.</h3>
        </div>
    """, unsafe_allow_html=True)

    # Inject custom CSS to style the navbar tabs
    st.markdown("""
        <style>
            /* Increase the font size of the tab titles */
            .stTabs [role="tab"] {
                font-size: 1.5em !important;
                font-weight: bold !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Navbar for navigation between models
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Deepfake Audio Detector",
         "Deepfake Image Detector",
         "Phishing Link Detector",
         "Extras"]
    )

    return tab1, tab2, tab3, tab4

def extras_tab(detector):
    st.title("AI Text Detector")
    st.write("Type your own text, upload a text file, or select from the sample texts to check if it is **Human-generated** or **AI-generated**.")

    sample_texts = [
        ("HUMAN-GENERATED", "Yesterday, I walked to the park to clear my head. The air was crisp, and the leaves crunched underfoot. I couldn’t help but feel a sense of nostalgia, remembering how my siblings and I used to play here when we were kids. It’s funny how places can hold so many memories."),
        ("AI-GENERATED", "The park represents a tranquil environment where individuals can contemplate life's deeper meaning and connect with the natural world. The gentle rustling of leaves, combined with the soft caress of a breeze, crafts an atmosphere of relaxation and mental clarity. Parks play a vital role in promoting social interaction, encouraging well-being, and maintaining the ecological balance of urban areas.")
    ]

    input_choice = st.radio("Choose input method:", ("Type text", "Upload text file", "Select sample text"))
    user_text = ""

    if input_choice == "Type text":
        user_text = st.text_area("Enter text:", height=450, placeholder="Start typing your text here...")
    elif input_choice == "Upload text file":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"], key="file_uploader_text")
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode("utf-8")
    elif input_choice == "Select sample text":
        selected_sample = st.selectbox("Select a sample text:", options=[f"{label}: {text[:50]}..." for label, text in sample_texts])
        if selected_sample:
            selected_label = selected_sample.split(": ", 1)[0]
            user_text = next(text for label, text in sample_texts if label == selected_label)
            st.text_area("Selected Sample Text:", value=user_text, height=200, disabled=True)

    if st.button("Classify Text"):
        if user_text.strip():
            st.write("Input text for classification:", user_text)
            try:
                prediction, human_prob, ai_prob = detector.classify_text(user_text)
                st.write("Classification result:", prediction, human_prob, ai_prob)
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
                        <h3 style="display: inline-block; margin-left: 30px;">
                            Prediction: <span style="color: {color};">{predicted_label}</span>
                        </h3>
                        <p style="display: inline-block; font-size: 20px; margin-left: -6px;">
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error during classification: {e}")
        else:
            st.error("Please enter, upload, or select some text to classify.")





def phishing_detection_navbar(phishing_detector):
    st.title("Phishing Detection")
    st.write("Enter a URL or select from the sample URLs, and the model will classify it as **SAFE** or **DANGEROUS**.")

    # Sample URLs
    sample_urls = [
        ("DANGEROUS", "https://míсrоsоft-update-check.net"),
        ("DANGEROUS", "https://fácebook-login.com"),
        ("DANGEROUS", "https://аpple-security-check.com"),
        ("SAFE", "https://www.facebook.com/"),
        ("SAFE", "https://www.microsoft.com/"),
        ("SAFE", "https://www.apple.com/"),
        ("SAFE", "https://www.omdena.com/")
    ]

    # Radio button for input method
    input_choice = st.radio("Choose input method:", ("Type your own URL", "Select a sample URL"))

    # Initialize the URL variable
    user_url = ""

    if input_choice == "Type your own URL":
        # Text input with placeholder
        user_url = st.text_input("Enter URL:", placeholder="https://www.enter-your-link-here.com")
    elif input_choice == "Select a sample URL":
        # Dropdown for sample URLs
        selected_sample = st.selectbox("Select a sample URL:", options=[f"{label}: {url}" for label, url in sample_urls])
        if selected_sample:
            # Extract the URL part from the selected option
            user_url = selected_sample.split(": ", 1)[1]
            # Display the selected URL in the text input for visualization
            st.text_input("Selected URL:", value=user_url, disabled=True)

    # Button to trigger prediction
    if st.button("Classify URL"):
        if user_url:  # Ensure there's a URL to classify
            # Get prediction and confidence
            label, confidence = phishing_detector.check_link_validity(user_url)

            # Display the result with centered text and color-coded prediction
            color = "green" if label == "SAFE" else "red"

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="display: inline-block; margin-left: 20px;">Prediction: <span style="color: {color};">{label}</span></h3>
                    <p style="display: inline-block; font-size: 20px; margin-left: -6px;">Confidence: {confidence*100:.2f}%</p>
                </div>
                """, 
                unsafe_allow_html=True)
        else:
            st.error("Please enter or select a URL to classify.")





def deepfake_audio_detector_menu(detector, test_files):
    st.title("Audio Deepfake Detector")
    st.write("Upload an audio file, or choose from the test files provided, and the AI will classify it as **Real** or **Fake**.")

    # Input choice: Upload or select a test file
    input_choice = st.radio("Choose input method:", ("Upload audio file", "Use test file"))

    selected_file_path = None

    if input_choice == "Upload audio file":
        uploaded_audio = st.file_uploader("Upload an Audio File", type=["wav", "mp3"], key="audio_upload")
        if uploaded_audio is not None:
            # Save uploaded audio temporarily
            selected_file_path = os.path.join("temp_uploaded_audio.mp3")
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_audio.read())
            st.audio(selected_file_path, format="audio/mp3", start_time=0)

    elif input_choice == "Use test file":
        # Display preloaded test files
        test_files_info = test_files["audio_files"]
        test_files_folder = test_files_info["folder"]
        test_files_list = test_files_info["files"]

        selected_test_file = st.selectbox("Select a test file:", test_files_list, key="audio_test_select")
        if selected_test_file:
            selected_file_path = os.path.join(test_files_folder, selected_test_file)
            st.audio(selected_file_path, format="audio/mp3", start_time=0)

    # Add unique keys for buttons
    classify_audio_key = "classify_audio_upload" if input_choice == "Upload audio file" else "classify_audio_test"

    if st.button("Classify Audio", key=classify_audio_key) and selected_file_path:
        try:
            predicted_label, confidence = detector.predict_audio_label(selected_file_path)
            # Determine color based on label
            color = "green" if predicted_label.upper() == "REAL" else "red"

            # Center the prediction and confidence output with original styling
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <h3 style="display: inline-block; margin-left: 30px;">
                        Prediction: <span style="color: {color};">{predicted_label}</span>
                    </h3>
                    <p style="display: inline-block; font-size: 20px; margin-left: -6px;">
                        Confidence: {confidence*100:.2f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"An error occurred while processing the audio file: {e}")



def deepfake_image_detector_menu(detector, test_files):
    st.title("Image Deepfake Detector")
    st.write("Upload an image file, or choose from the test files provided, and the AI will classify it as **Real** or **Fake**.")

    # Input choice: Upload or select a test file
    input_choice = st.radio("Choose input method:", ("Upload image file", "Use test file"))

    selected_file_path = None

    if input_choice == "Upload image file":
        uploaded_image = st.file_uploader("Upload an Image File", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            # Save uploaded image temporarily
            selected_file_path = os.path.join("temp_uploaded_image.jpg")
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_image.read())

            # Display the uploaded image with fixed width
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{image_to_base64(Image.open(selected_file_path))}" alt="Uploaded Image" width="400"/>
                    <p><strong>Uploaded Image</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif input_choice == "Use test file":
        # Use preloaded test files
        folder_path = test_files["image_files"]["folder"]
        test_files_list = test_files["image_files"]["files"]
        selected_test_file = st.selectbox("Select a test file:", test_files_list)
        if selected_test_file:
            selected_file_path = os.path.join(folder_path, selected_test_file)
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{image_to_base64(Image.open(selected_file_path))}" alt="Test Image" width="400"/>
                    <p><strong>Selected Test Image</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Classify Image"):
        if selected_file_path:
            try:
                # Open the image file
                image = Image.open(selected_file_path)

                # Pass the image object to the detector
                predicted_label, confidence = detector.predict(image)

                # Determine color based on label
                color = "green" if predicted_label.upper() == "REAL" else "red"

                # Center the prediction and confidence output with original styling
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <h3 style="display: inline-block; margin-left: 30px;">
                            Prediction: <span style="color: {color};">{predicted_label}</span>
                        </h3>
                        <p style="display: inline-block; font-size: 20px; margin-left: -6px;">
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"An error occurred while processing the image file: {e}")
        else:
            st.error("Please select or upload an image file to classify.")




# Function to convert the image to base64 for inline display in HTML
def image_to_base64(image):
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@st.cache_resource
def load_phishing_detector():
    import os
    model_path = os.path.join("models", "phishing_detection")
    return helpers.PhishingDetector(model_path)


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
    model_path = os.path.join("models", "ai_text_detector", "ai_text_detector_model.pkl")

    # Verify the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Ensure the file is properly downloaded and placed in the correct directory."
        )

    print(f"Loading text detector model from {model_path}...")
    return helpers.AITextDetector(model_path=model_path)

@st.cache_resource
def download_test_files(folder_id, local_folder_name):
    """Download test files from Google Drive."""
    folder_path = os.path.join("models", local_folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Downloading {local_folder_name} test files from Google Drive...")
        gdown.download_folder(id=folder_id, output=folder_path, quiet=False)
    return folder_path

@st.cache_resource
def preload_test_files():
    """Preload test files from Google Drive."""
    # Audio test files
    audio_folder = download_test_files(AUDIO_TEST_FILES_FOLDER_ID, "audio_test_files")
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3"))]

    # Image test files
    image_folder = download_test_files(IMAGE_TEST_FILES_FOLDER_ID, "image_test_files")
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    return {
        "audio_files": {"folder": audio_folder, "files": audio_files},
        "image_files": {"folder": image_folder, "files": image_files},
    }


def main():
    # Initialize Streamlit app
    init_streamlit()

    # Download models if they don't exist locally
    download_models(MODEL_FOLDER_ID)

    # Preload test files
    test_files = preload_test_files()

    # Display navbar and tabs
    tab1, tab2, tab3, tab4 = display_navbar()

    # Load the models using the caching functions (do this after downloading)
    audio_detector = load_audio_detector()
    image_detector = load_image_detector()
    text_detector = load_text_detector()
    phishing_detector = load_phishing_detector()

    with tab1:
        deepfake_audio_detector_menu(audio_detector, test_files)
    with tab2:
        deepfake_image_detector_menu(image_detector, test_files)
    with tab3:
        phishing_detection_navbar(phishing_detector)
    with tab4:
        extras_tab(text_detector)

if __name__ == "__main__":
    main()