# ai_model.py

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
from deepface import DeepFace
from openai import OpenAI  # Import OpenAI class
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.linear_model import LinearRegression
from models import Rating
from app import db, app
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
azure_api_key = os.getenv("AZURE_COMPUTER_VISION_KEY")
azure_endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")

# Instantiate the OpenAI client
client = OpenAI(api_key=api_key)

# Initialize the Computer Vision client
computervision_client = ComputerVisionClient(
    azure_endpoint, CognitiveServicesCredentials(azure_api_key)
)

# Load the pre-trained model once
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(img_path):
    """Extract features from an image using MobileNetV2."""
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = mobilenet_model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None  # Return None to indicate failure


def detect_gender(img_path):
    """Detect gender using DeepFace with enforced detection set to False."""
    try:
        analysis = DeepFace.analyze(img_path, actions=['gender'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        gender_data = analysis.get('gender', {})
        return 'man' if gender_data.get('Man', 0) > gender_data.get('Woman', 0) else 'woman'
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        return None

def train_user_model(user_id):
    """Train a regression model for the user based on their ratings."""
    ratings = Rating.query.filter_by(user_id=user_id).all()
    if not ratings:
        print("No ratings found for user.")
        return

    X = []
    y = []
    for rating in ratings:
        img_path = os.path.join(app.static_folder, rating.image_filename.lstrip('/'))
        if os.path.isdir(img_path) or not os.path.isfile(img_path):
            print(f"Invalid image file: {img_path}")
            continue

        features = extract_image_features(img_path)
        if features is not None:
            X.append(features)
            y.append(rating.rating)
        else:
            print(f"Skipping image due to previous error: {img_path}")
    if not X:
        print("No valid images for training.")
        return

    X = np.array(X)
    y = np.array(y)
    
    print(f"Training model for user {user_id}")
    print(f"Number of samples: {len(y)}")
    print(f"Features shape: {X.shape}")
    print(f"Ratings: {y}")

    # Dynamically set the number of PCA components
    n_components = min(100, len(X), X.shape[1])  # Choose the smallest value among 100, samples, and features

    if n_components < 1:
        print("Not enough data to perform PCA.")
        return

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Use Ridge Regression
    model = Ridge(alpha=1.0)
    model.fit(X_pca, y)
    
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")

    # Save both the PCA and the model
    model_data = {'pca': pca, 'model': model}

    # Ensure the model path is correct and overwrite existing model
    model_dir = os.path.join(app.root_path, 'user_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'model_{user_id}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model trained and saved for user {user_id}.")

def process_images(image_paths, user):
    """Process uploaded images to predict rating and generate a pickup line."""
    model_path = os.path.join(app.root_path, 'user_models', f'model_{user.id}.pkl')
    if not os.path.exists(model_path):
        return {'success': False, 'message': 'User model not found. Please recalibrate.'}

    # Load the user-specific model and PCA transformer
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Check if model_data is a dictionary
    if isinstance(model_data, dict) and 'model' in model_data and 'pca' in model_data:
        model = model_data['model']
        pca = model_data['pca']
    else:
        # Handle old model format
        print("Loaded model data is not in the expected format. Please recalibrate.")
        return {'success': False, 'message': 'Model format error. Please recalibrate.'}

    # Extract features for each uploaded image
    features = []
    valid_image_paths = []

    for img_path in image_paths:
        feat = extract_image_features(img_path)
        if feat is not None:
            features.append(feat)
            valid_image_paths.append(img_path)
        else:
            print(f"Skipping image due to error: {img_path}")

    if not features:
        print("No valid images to process.")
        return {'success': False, 'message': 'Could not extract features from any uploaded images.'}

    # Convert features to numpy array
    X = np.array(features)

    # Apply PCA transformation
    X_pca = pca.transform(X)

    # Predict individual ratings based on features
    ratings = model.predict(X_pca)
    print(f"Predicted ratings: {ratings}")

    # Calculate average rating rounded to nearest 0.5
    average_rating = round(np.mean(ratings) * 2) / 2

    # Generate pickup line based on the images and user preference
    pickup_line = generate_pickup_line(valid_image_paths, user)

    return {
        'success': True,
        'ratings': ratings,          # Individual ratings for each image
        'average_rating': average_rating,
        'pickup_line': pickup_line   # Generated pickup line based on image descriptions and preference
    }

def describe_image(img_path):
    """Generate a detailed description of the image using Microsoft Computer Vision."""
    try:
        # Open the image file
        with open(img_path, 'rb') as image_stream:
            # Call the API to get a description
            description_results = computervision_client.describe_image_in_stream(image_stream)
        if description_results.captions:
            # Get the most confident caption
            caption = max(description_results.captions, key=lambda c: c.confidence)
            return caption.text
        else:
            return "No description available."
    except Exception as e:
        print(f"Error describing image {img_path}: {e}")
        return "Error in generating description."

def generate_pickup_line(image_paths, user):
    """Generate a pickup line based on detailed analysis of uploaded photos."""
    descriptions = [describe_image(img_path) for img_path in image_paths]
    background_info = "\n".join(descriptions)
    if len(image_paths) > 1:
        prompt_for_selection = (
            "Given the following descriptions of photos:\n\n"
            f"{background_info}\n\n"
            "Select the most intriguing description that could inspire a catchy pickup line."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that selects the best image description for a pickup line."},
                {"role": "user", "content": prompt_for_selection}
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.7,
            stop=['Human:', 'AI:']
        )
        selected_description = response.choices[0].message.content.strip()
    else:
        selected_description = descriptions[0]
    final_prompt = (
        f"Create a catchy, inviting pickup line for a person interested in a {user.preference} "
        f"based on the following photo description: '{selected_description}'. "
        "The pickup line must be a maximum of 20 words, preferably shorter."
    )
    pickup_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant that generates witty pickup lines."},
            {"role": "user", "content": final_prompt}
        ],
        max_tokens=50,
        temperature=0.9,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.7,
        stop=['Human:', 'AI:']
    )
    return pickup_response.choices[0].message.content.strip()
