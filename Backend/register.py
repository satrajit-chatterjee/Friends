from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import random, sys
app = Flask(__name__)

app.config['SECRET_KEY'] = 'a05e08fbc2d904a43692e593a0f04431'  # to be kept secret
app.config['SQLALCHEMY_DATABASE_URI'] = r'sqlite:///site.db'   # to tell the program that a file named site.db exists on
#  the relative path
db = SQLAlchemy(app)


class User(db.Model):
    name = db.Column(db.String(30), nullable=False)
    contact = db.Column(db.Integer(), nullable=False)
    address = db.Column(db.Text, nullable=False)
    interests = db.Column(db.Text, nullable=True)
    imei = db.Column(db.Integer(), unique=True, nullable=False, primary_key=True)
    profile_image = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return self.name


'''
if program is run for the first time, i.e. site.db is not created yet, 
    enter create as an argument when running this app from the cmd
'''

try:
    if sys.argv[1] == 'create':
        from register import db
        db.create_all()
except:
    pass


@app.route('/register', methods = ['POST'])
def register():
    name = request.json['name']
    contact = request.json['contact']
    address = request.json['address']
    interests = request.json['interests']
    imei = request.json['imei']
    profile_image = request.json['profile_image']
    new_user = User(name=name, contact=contact, address=address, interests=interests, imei=imei, profile_image=profile_image)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'otp': random.randint(1000, 9999)})


if __name__ == '__main__':
    app.run(debug = True)