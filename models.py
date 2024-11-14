# models.py
from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime

# This function is used by Flask-Login to load the current user
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    preference = db.Column(db.String(10), nullable=False)  # 'man' or 'woman'
    trial_start = db.Column(db.DateTime, default=datetime.utcnow)
    is_subscribed = db.Column(db.Boolean, default=True)  # Default to True for this setup
    ratings = db.relationship('Rating', backref='user', lazy=True)  # Establish relationship to Rating

class Rating(db.Model):
    __tablename__ = 'rating'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
