# app.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

from routes import *

if __name__ == '__main__':
    app.run(debug=True)
