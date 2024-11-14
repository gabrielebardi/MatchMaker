to activate the environment
source venv/bin/activate 

to start the DB
python
from app import app, db
with app.app_context():
    db.create_all()


to start the app in a new termnial tab
flask run
OR
python app.py

