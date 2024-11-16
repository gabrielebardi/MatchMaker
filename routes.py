# routes.py
from flask import render_template, url_for, flash, redirect, request, abort
from app import app, db
from models import User, Rating
from forms import RegistrationForm, LoginForm
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from ai_model import detect_gender, process_images, train_user_model
import os
import random

# Authentication Routes

@app.route('/test_auth')
def test_auth():
    print("Current user authenticated:", current_user.is_authenticated)
    return "Check the terminal for authentication status"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        print("User authenticated:", current_user.is_authenticated)
        return redirect(url_for('home'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(email=form.email.data, password=hashed_password, preference=form.preference.data)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash('Account created! Please complete the calibration.', 'success')
        return redirect(url_for('calibrate'))
    else:
        print(form.errors)
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        print("User authenticated:", current_user.is_authenticated)
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Check email and password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Home Route

@app.route('/')
@app.route('/home')
@login_required
def home():
    return render_template('home.html')

# Calibration Route
@app.route('/calibrate', methods=['GET', 'POST'])
@login_required
def calibrate():
    if request.method == 'POST':
        # Delete all existing ratings for the current user
        Rating.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        
        # Retrieve form data
        ratings = []
        image_filenames = []
        
        # Get all keys starting with 'image_filename_'
        image_ids = [key.split('_')[-1] for key in request.form.keys() if key.startswith('image_filename_')]
        
        for image_id in image_ids:
            rating = request.form.get(f'rating_{image_id}')
            image_filename = request.form.get(f'image_filename_{image_id}')
            if rating and image_filename:
                ratings.append(rating)
                image_filenames.append(image_filename)
            else:
                print(f"Missing data for image_id {image_id}: rating={rating}, image_filename={image_filename}")
        
        # Debugging statements
        print(f"Received ratings: {ratings}")
        print(f"Received image filenames: {image_filenames}")

        if not ratings or not image_filenames:
            print("No ratings or image filenames received.")
        
        # Save new ratings
        for filename, rating in zip(image_filenames, ratings):
            print(f"Saving rating {rating} for image {filename}")
            rating_entry = Rating(
                user_id=current_user.id,
                image_filename=filename,
                rating=int(rating)
            )
            db.session.add(rating_entry)
        db.session.commit()
        
        # Confirm new ratings are saved
        num_ratings = Rating.query.filter_by(user_id=current_user.id).count()
        print(f"Number of ratings after saving new ratings: {num_ratings}")
        
        # Train the AI model for the user based on new ratings
        train_user_model(current_user.id)
        flash('Calibration complete!', 'success')
        return redirect(url_for('home'))

    # Handle GET request: Display unique images for calibration based on user's preference
    gender = current_user.preference.lower()

    # Define directories for AI and celebrity images based on gender
    ai_images_dir = os.path.join(app.static_folder, 'images', 'calibration', 'ai', gender)
    celeb_images_dir = os.path.join(app.static_folder, 'images', 'calibration', 'celeb', gender)

    # Retrieve unique image paths, filtering for specific formats
    ai_images = list(set([
        os.path.join('images', 'calibration', 'ai', gender, filename)
        for filename in os.listdir(ai_images_dir)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]))

    celeb_images = list(set([
        os.path.join('images', 'calibration', 'celeb', gender, filename)
        for filename in os.listdir(celeb_images_dir)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]))

    # Combine images and select 15 unique images from each folder for a total of 30
    combined_images = random.sample(ai_images, min(15, len(ai_images))) + random.sample(celeb_images, min(15, len(celeb_images)))
    
    # Shuffle to ensure a random order
    random.shuffle(combined_images)

    # Assign a unique ID to each image and structure for display
    image_list = [{'id': idx, 'filename': img, 'source': 'AI' if 'ai' in img else 'Celebrity'} for idx, img in enumerate(combined_images)]

    return render_template('calibrate.html', images=image_list)
    

# Re-Calibration Route - confirm gender
@app.route('/confirm_gender', methods=['GET', 'POST'])
@login_required
def confirm_gender():
    if request.method == 'POST':
        # Get the selected gender from the form
        gender = request.form.get('gender')
        if gender in ['man', 'woman']:
            # Update user's preference in the database if needed
            current_user.preference = gender
            db.session.commit()
            flash('Gender preference updated. Starting calibration...', 'success')
            return redirect(url_for('calibrate'))  # Go to the calibration route
        else:
            flash('Please select a valid gender.', 'danger')
    
    return render_template('confirm_gender.html')

# Evaluation Route
@app.route('/evaluate', methods=['GET', 'POST'])
@login_required
def evaluate():
    if request.method == 'POST':
        files = request.files.getlist('images')
        
        if not files:
            flash('No files uploaded', 'error')
            return redirect(url_for('evaluate'))

        uploads_dir = os.path.join(app.static_folder, 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        saved_paths = []
        for file in files:
            save_path = os.path.join(uploads_dir, file.filename)
            file.save(save_path)

            detected_gender = detect_gender(save_path)
            if detected_gender != current_user.preference:
                flash("One or more images do not match your gender preference.", 'error')
                return redirect(url_for('evaluate'))

            saved_paths.append(save_path)

        # Process images and retrieve average rating and pickup line
        result = process_images(saved_paths, current_user)
        if not result['success']:
            flash(result['message'], 'error')
            return redirect(url_for('evaluate'))

        # Pass 'zip' into the template context
        return render_template('result.html', result=result, images=saved_paths, zip=zip)

    return render_template('evaluate.html')


# Profile Route for Subscription Info
@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', subscribed=current_user.is_subscribed)

# Error Handlers

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404