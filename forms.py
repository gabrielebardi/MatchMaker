# forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, RadioField
from wtforms.validators import DataRequired, Email, EqualTo

class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    preference = RadioField('Preference', choices=[('man', 'Man'), ('woman', 'Woman')], validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
