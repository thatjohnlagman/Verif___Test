import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class DeepfakeImageDetector:
    def __init__(self, model_dir, label_map):
        config = AutoConfig.from_pretrained(model_dir)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_config(config)
        self.label_map = label_map

        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_image(self, image):
        image = image.convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs

    def predict(self, image):
        inputs = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()
            predicted_label_id = logits.argmax(dim=-1).item()
            predicted_label = self.label_map[predicted_label_id]
        return predicted_label, probabilities[predicted_label_id]

class AITextDetector:
    def __init__(self, model_dir, max_length=128):
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer and model from the local directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def classify_text(self, text):
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move tensors to the appropriate device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

        # probs is now an array of probabilities for each class.
        human_prob = probs[0]
        ai_prob = probs[1]

        # Return the predicted class and probabilities
        return (0 if ai_prob < 0.5 else 1), human_prob, ai_prob

class DeepfakeAudioDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def audio_to_spectrogram(self, audio_path, output_path, spectrogram_size=(224, 224), dpi=300):
        """
        Converts an audio file to a Mel spectrogram with enhanced visual characteristics.
        """
        # Load audio file using librosa with soundfile as the backend
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Generate Mel spectrogram
        n_fft = 2048
        hop_length = 256
        n_mels = 128

        spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Convert to decibel scale and normalize
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = (spectrogram_db + 80) / 80  # Normalize to 0-1 range

        # Create a figure and plot the spectrogram
        plt.figure(figsize=(8, 6), dpi=dpi)
        plt.axis('off')
        librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            hop_length=hop_length,
            cmap='inferno'
        )

        # Save the spectrogram
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

        # Resize the saved image to target dimensions
        img = Image.open(output_path)
        img = img.resize(spectrogram_size, Image.LANCZOS)
        img.save(output_path)

    def preprocess_image(self, image_path, target_size=(224, 224)):
        # Load and preprocess the spectrogram image
        img = load_img(image_path, target_size=target_size)
        img = img.convert('RGB')  # Ensure RGB mode
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def predict_audio_label(self, audio_path, target_size=(224, 224), spectrogram_path='temp_spectrogram.png'):
        # Convert audio to spectrogram
        self.audio_to_spectrogram(audio_path, spectrogram_path, spectrogram_size=target_size)

        # Preprocess the spectrogram image
        image_array = self.preprocess_image(spectrogram_path, target_size=target_size)

        # Make a prediction
        prediction = self.model.predict(image_array)

        # Interpret the result
        label = "REAL" if prediction[0][0] > 0.5 else "FAKE"
        confidence = prediction[0][0] if label == "REAL" else 1 - prediction[0][0]
        return label, confidence

class PhishingDetector:
    def __init__(self, model_path, tokenizer_path):
        # Load the tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

        # Load the model configuration
        config = DistilBertConfig.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_config(config)

        # Load the model's state_dict
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def check_link_validity(self, link):
        # Tokenize the input link text
        inputs = self.tokenizer(link, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get predictions from the model
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the predicted class and confidence
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

        # Map the predicted class to its label (BENIGN or MALWARE)
        id2label = {0: "BENIGN", 1: "MALWARE"}
        label = id2label[predicted_class]

        return label, confidence