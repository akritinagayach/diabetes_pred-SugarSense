import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from werkzeug.security import check_password_hash
import json
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import requests
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# Initialize Flask app
app = Flask(__name__)
import os
app.secret_key = os.environ.get("32f03892e75254eee54a9921d7de32cea3e29f7eb6688a279b2703c71f84b313", 'default-key-for-dev')

# Function to load or create XGBoost model
def load_or_create_model():
    try:
        model = pickle.load(open("diabetes_model.pkl", "rb"))
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Recreating the model...")
        try:
            data = pd.read_csv("modeldataset.csv")
            X = data.drop("Diabetes_012", axis=1)
            y = data["Diabetes_012"]
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X, y)
            pickle.dump(model, open("diabetes_model.pkl", "wb"))
            print("Model created and saved successfully!")
            return model
        except Exception as e:
            print(f"Error creating model: {e}")
            return None

# Load the XGBoost model
model = load_or_create_model()

# Home route
@app.route('/')
def index():
    return render_template('index.html')


import re

# Email validation function
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not (name and email and password and confirm_password):
            flash('Please fill out all fields.', 'danger')
            return redirect(url_for('register'))

        if not is_valid_email(email):
            flash('Invalid email format! Please enter a valid email.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (name, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email or username already exists.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Please enter valid credentials.', 'danger')
            return redirect(url_for('login'))

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM users WHERE email = ? ', (email,))
        user = cursor.fetchone()
        conn.close()

        if user:
            print(f"User found: {user}")  # Debugging
            stored_password = user[1]
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))    

        if check_password_hash(stored_password, password):
            session['user_id'] = user[0]
            flash('Login successful!', 'success')
            return redirect(url_for('mainpage2'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Main page route
@app.route('/mainpage2')
def mainpage2():
    if 'user_id' not in session:
        flash('Please log in first.', 'danger')
        return redirect(url_for('login'))
    return render_template('mainpage2.html')
# Prediction route


# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            genhlth = int(request.form.get('general_health'))
            highbp = int(request.form.get('blood_pressure'))
            bmi = float(request.form.get('bmi'))
            diffwalk = int(request.form.get('diffwalk'))
            highchol = int(request.form.get('highchol'))
            age = int(request.form.get('age'))
            heartdisease = int(request.form.get('heartdisease'))
            physhealth = int(request.form.get('physical_health'))
            stroke = int(request.form.get('stroke'))
            menthealth = int(request.form.get('mental_health'))
            cholcheck = int(request.form.get('cholcheck'))
            smoker = int(request.form.get('smoking'))

            input_features = np.array([[genhlth, highbp, bmi, diffwalk, highchol, age, heartdisease, physhealth, stroke, menthealth, cholcheck, smoker]])
            prediction = model.predict(input_features)
            result = "Diabetic" if prediction[0] == 2 else "Prediabetic" if prediction[0] == 1 else "Non-Diabetic"

            conn = sqlite3.connect('predict.db')
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO predictions (user_id, genhlth, highbp, bmi, diffwalk, highchol, age, heartdisease, physhealth, stroke, menthealth, cholcheck, smoker, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (session.get('user_id'), genhlth, highbp, bmi, diffwalk, highchol, age, heartdisease, physhealth, stroke, menthealth, cholcheck, smoker, result))
            conn.commit()
            conn.close()

            return render_template('predict.html', prediction_text=result)
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html')
# Other routes...
@app.route('/meal', methods=['GET', 'POST'])
def meal():
    

    return render_template('meal.html')




# Function to get prediction result
def get_prediction_result(user_id):
    try:
        conn = sqlite3.connect("predict.db")
        cursor = conn.cursor()
        cursor.execute("SELECT prediction FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            print(f"✅ Prediction for user {user_id}: {result[0]}")
            return result[0]  # Ensuring it's a string
        else:
            print("⚠️ No prediction found!")
            return None
    except Exception as e:
        print(f"❌ Error fetching prediction result: {str(e)}")
        return None

# API endpoint to fetch latest prediction result
@app.route('/get_prediction_result', methods=['GET'])
def fetch_prediction_result():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    result = get_prediction_result(user_id)
    if not result:
        return jsonify({"error": "No prediction found"}), 404
    
    return jsonify({"prediction_result": result}) if result else jsonify({"error": "No prediction found"}), 404



def calculate_bmi(height, weight):
    try:
        height = float(height)  # Ensure height is a float
        weight = float(weight)  # Ensure weight is a float
        
        if height <= 0 or weight <= 0:
            raise ValueError("Height and weight must be positive numbers.")

        bmi = round(weight / ((height / 100) ** 2), 1)  # BMI formula
        print(f"🔍 Calculated BMI: {bmi} (Height: {height} cm, Weight: {weight} kg)")  # Debugging print
        return bmi
    except Exception as e:
        print(f"❌ Error calculating BMI: {e}")
        return None  # Return None if there's an error

def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 24.9:
        return "Normal"
    elif 25 <= bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

def get_age_group(age):
    age = int(age)
    if age <= 25:
        return "18-25"
    elif 26 <= age <= 35:
        return "26-35"
    elif 36 <= age <= 45:
        return "36-45"
    elif 46 <= age <= 60:
        return "46-60"
    else:
        return "60+"




# Function to get prediction result
def get_prediction_result(user_id):
    try:
        conn = sqlite3.connect("predict.db")
        cursor = conn.cursor()
        cursor.execute("SELECT prediction FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            print(f"✅ Prediction for user {user_id}: {result[0]}")
            return result[0]  # Ensuring it's a string
        else:
            print("⚠️ No prediction found!")
            return None
    except Exception as e:
        print(f"❌ Error fetching prediction result: {str(e)}")
        return None


@app.route('/excercise', methods=['GET', 'POST'])
def excercise():
    if 'user_id' not in session:
        flash('Please log in first.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    prediction_result = get_prediction_result(user_id)
    
    if not prediction_result:
        flash("No prediction found. Please complete the prediction first.", "danger")
        return redirect(url_for('predict'))

    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        height = request.form.get('height')
        weight = request.form.get('weight')

        if not all([age, gender, height, weight]):
            flash("Please enter all required details.", "danger")
            return render_template('excercise.html', age=age, gender=gender, height=height, weight=weight)

        age = int(age)
        height = float(height)
        weight = float(weight)

        bmi = calculate_bmi(height, weight)
        bmi_category = categorize_bmi(bmi)
        
        
        prediction_result = get_prediction_result(user_id)
        if not prediction_result:
            flash("No prediction found. Please complete the prediction first.", "danger")
            return redirect(url_for('predict'))
        # Pass correct parameters
        exercise_plan = get_exercise_plan(user_id, age, bmi_category, gender, prediction_result)

        return render_template('excercise.html', exercise_plan=exercise_plan, age=age, gender=gender, height=height, weight=weight)
    
    return render_template('excercise.html')

def get_exercise_plan(user_id, age, bmi_category, gender, prediction_result):
    """Returns a personalized exercise plan based on user details"""

    print(age,bmi_category, gender, prediction_result)

    # Determine age group
    if 18 <= age <= 25:
        age_group = "18-25"
    elif 26 <= age <= 35:
        age_group = "26-35"
    elif 36 <= age <= 45:
        age_group = "36-45"
    elif 46 <= age <= 55:
        age_group = "46-55"
    else:
        age_group = "56+"
        
    
    workout_type = "Walking & Stretching"
    frequency = "Daily"
    duration = "30 minutes"
    exercises = "Brisk Walking, Gentle Yoga"

    # Conditions based on age, BMI, gender, and diabetes prediction
    if age_group == "18-25":
        if bmi_category == "Underweight":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Strength Training"
                    frequency = "3 times a week"
                    duration = "45 minutes"
                    exercises = "Push-ups, Squats, Deadlifts"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Strength & Cardio"
                    frequency = "4 times a week"
                    duration = "50 minutes"
                    exercises = "Bodyweight Training, Cycling"
                else:  # Diabetes
                    workout_type = "Light Strength & Walking"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Brisk Walking, Resistance Bands"

            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Strength & Yoga"
                    frequency = "3 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Light Weights, Pilates"
                elif prediction_result == "Pre-diabetic":
                    workout_type = "Yoga & Bodyweight"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Planks, Resistance Bands"
                else:  # Diabetes
                    workout_type = "Yoga & Light Cardio"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Walking, Stretching"

        elif bmi_category == "Normal":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Cardio & Strength"
                    frequency = "5 times a week"
                    duration = "50 minutes"
                    exercises = "Running, Push-ups, Pull-ups"
                elif prediction_result == "Pre-diabetic":
                    workout_type = "HIIT & Strength"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Jump Squats, Kettlebell Swings"
                else:  # Diabetes
                    workout_type = "Low-Impact Strength"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Brisk Walking, Dumbbells"

        elif bmi_category == "Overweight":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Weight Loss Cardio"
                    frequency = "5 times a week"
                    duration = "60 minutes"
                    exercises = "Cycling, Jump Rope, Swimming"
                elif prediction_result == "Pre-diabetic":
                    workout_type = "Low-Impact Strength"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Rowing, Strength Bands"
                else:  # Diabetes
                    workout_type = "Controlled Movement"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Tai Chi, Swimming"
                    
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Cardio & Strength"
                    frequency = "4-5 times a week"
                    duration = "45 minutes"
                    exercises = "Jogging, Squats, Dumbbell Workouts"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Strength & Endurance"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Bodyweight Exercises, Resistance Bands"
                else:  # Diabetes
                    workout_type = "Low-Impact Cardio & Yoga"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Walking, Yoga, Light Resistance Training"

        elif bmi_category == "Obese":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low Impact & Strength"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Rowing, Resistance Training"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Light Cardio"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Walking, Swimming"
                else:  # Diabetes
                    workout_type = "Senior Fitness & Walking"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Stretching, Light Weights, Walking"
                    
        elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low-Impact Cardio & Strength"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Swimming, Resistance Bands"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Gentle Cardio & Yoga"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Water Aerobics, Light Resistance Training"
                else:  # Diabetes
                    workout_type = "Mobility & Low-Impact Fitness"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Chair Exercises, Light Stretching, Walking"
                    
    elif age_group == "26-35":
        if bmi_category == "Underweight":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Muscle Gain & Strength"
                    frequency = "4 times a week"
                    duration = "50 minutes"
                    exercises = "Deadlifts, Bench Press, Squats"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Strength & Moderate Cardio"
                    frequency = "4 times a week"
                    duration = "50 minutes"
                    exercises = "Cycling, Dumbbell Workouts"
                else:  # Diabetes
                    workout_type = "Light Strength & Endurance"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Brisk Walking, Resistance Bands"

            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Yoga & Strength"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Pilates, Resistance Bands, Planks"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Strength & Flexibility"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Yoga, Bodyweight Workouts"
                else:  # Diabetes
                    workout_type = "Yoga & Light Endurance"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Walking, Light Resistance Training"
                    
        elif bmi_category == "Normal":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Functional Strength & Cardio"
                    frequency = "5 times a week"
                    duration = "50 minutes"
                    exercises = "Deadlifts, Running, Pull-ups"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Moderate HIIT & Strength"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Kettlebell Swings, Jump Squats"
                else:  # Diabetes
                    workout_type = "Low-Impact Strength & Endurance"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Brisk Walking, Dumbbells, Yoga"

            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Cardio & Functional Strength"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Jogging, Bodyweight Workouts, Pilates"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Balanced Strength & Endurance"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Yoga, Resistance Bands, Cycling"
                else:  # Diabetes
                    workout_type = "Low-Impact Cardio & Core Strength"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Walking, Core Workouts, Stretching"
                    
        elif bmi_category == "Obese":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low Impact & Strength"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Rowing, Resistance Training"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Light Cardio"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Walking, Swimming"
                else:  # Diabetes
                    workout_type = "Senior Fitness & Walking"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Stretching, Light Weights, Walking"
                    
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low-Impact Cardio & Strength"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Swimming, Resistance Bands"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Gentle Cardio & Yoga"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Water Aerobics, Light Resistance Training"
                else:  # Diabetes
                    workout_type = "Mobility & Low-Impact Fitness"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Chair Exercises, Light Stretching, Walking"
                    
    elif age_group == "36-50":
        if bmi_category == "Underweight":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Moderate Strength Training"
                    frequency = "3-4 times a week"
                    duration = "45 minutes"
                    exercises = "Bodyweight Squats, Resistance Bands, Light Dumbbells"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Strength & Low-Impact Cardio"
                    frequency = "4 times a week"
                    duration = "50 minutes"
                    exercises = "Cycling, Resistance Band Workouts"
                else:  # Diabetes
                    workout_type = "Endurance & Balance Training"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Walking, Light Strength Work"
                    
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Strength & Flexibility"
                    frequency = "3-4 times a week"
                    duration = "40 minutes"
                    exercises = "Pilates, Resistance Bands, Yoga"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Gentle Strength & Cardio"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Planks, Cycling, Resistance Bands"
                else:  # Diabetes
                    workout_type = "Mobility & Light Strength"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Stretching, Light Weights, Walking"  
                    
        if bmi_category == "Normal":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Balanced Strength & Cardio"
                    frequency = "4-5 times a week"
                    duration = "45 minutes"
                    exercises = "Jogging, Bodyweight Workouts, Resistance Training"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Low-Impact HIIT & Strength"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Swimming, Resistance Band Training"
                else:  # Diabetes
                    workout_type = "Controlled Strength & Cardio"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Brisk Walking, Light Dumbbell Exercises"
                    
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Functional Strength & Flexibility"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Pilates, Resistance Bands, Yoga"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Moderate Cardio & Strength"
                    frequency = "4 times a week"
                    duration = "45 minutes"
                    exercises = "Cycling, Planks, Resistance Bands"
                else:  # Diabetes
                    workout_type = "Mobility & Light Strength"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Stretching, Light Weights, Walking"    
                    
        elif bmi_category == "Obese":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low-Impact Strength & Cardio"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Rowing, Resistance Training"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Gentle Cardio & Flexibility"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Walking, Swimming"
                else:  # Diabetes
                    workout_type = "Senior Fitness & Walking"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Stretching, Light Weights, Walking"
                    
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low-Impact Cardio & Strength"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Swimming, Resistance Bands"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Gentle Cardio & Yoga"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Water Aerobics, Light Resistance Training"
                else:  # Diabetes
                    workout_type = "Mobility & Low-Impact Fitness"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Chair Exercises, Light Stretching, Walking"  
                    
                    
    elif age_group == "50+":
        if bmi_category == "Underweight":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Senior Strength & Mobility"
                    frequency = "3 times a week"
                    duration = "40 minutes"
                    exercises = "Light Dumbbell Exercises, Chair Yoga, Stretching"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Low-Impact Strength & Cardio"
                    frequency = "4 times a week"
                    duration = "40 minutes"
                    exercises = "Brisk Walking, Resistance Bands, Water Aerobics"
                else:  # Diabetes
                    workout_type = "Gentle Mobility & Cardio"
                    frequency = "5 times a week"
                    duration = "35 minutes"
                    exercises = "Chair Exercises, Light Resistance Training, Walking"
            
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Strength & Flexibility"
                    frequency = "3 times a week"
                    duration = "40 minutes"
                    exercises = "Pilates, Yoga, Resistance Bands"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Low-Impact Strength & Cardio"
                    frequency = "4 times a week"
                    duration = "40 minutes"
                    exercises = "Water Aerobics, Brisk Walking, Stretching"
                else:  # Diabetes
                    workout_type = "Gentle Mobility & Flexibility"
                    frequency = "5 times a week"
                    duration = "35 minutes"
                    exercises = "Chair Yoga, Light Stretching, Walking"
                    
        elif bmi_category == "Normal":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Balanced Cardio & Strength"
                    frequency = "4-5 times a week"
                    duration = "45 minutes"
                    exercises = "Swimming, Weight Training, Brisk Walking"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Low-Impact Strength & Endurance"
                    frequency = "4 times a week"
                    duration = "40 minutes"
                    exercises = "Resistance Bands, Stationary Cycling"
                else:  # Diabetes
                    workout_type = "Gentle Movement & Flexibility"
                    frequency = "5 times a week"
                    duration = "35 minutes"
                    exercises = "Tai Chi, Chair Exercises, Walking"
            
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Light Cardio & Strength"
                    frequency = "4-5 times a week"
                    duration = "40 minutes"
                    exercises = "Pilates, Brisk Walking, Water Aerobics"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Flexibility & Strength Training"
                    frequency = "4 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Resistance Bands, Light Dumbbells"
                else:  # Diabetes
                    workout_type = "Senior Mobility & Cardio"
                    frequency = "5 times a week"
                    duration = "35 minutes"
                    exercises = "Chair Yoga, Stretching, Walking"
                    
        elif bmi_category == "Obese":
            if gender == "male":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Low-Impact Strength & Cardio"
                    frequency = "5 times a week"
                    duration = "45 minutes"
                    exercises = "Brisk Walking, Water Aerobics, Resistance Bands"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Gentle Movement & Endurance"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Tai Chi, Swimming, Stretching"
                else:  # Diabetes
                    workout_type = "Senior Fitness & Flexibility"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Chair Yoga, Light Resistance, Walking"
            
            elif gender == "female":
                if prediction_result == "Non-Diabetic":
                    workout_type = "Light Strength & Flexibility"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Pilates, Brisk Walking, Water Aerobics"
                elif prediction_result == "Pre-Diabetic":
                    workout_type = "Low-Impact Strength & Cardio"
                    frequency = "5 times a week"
                    duration = "40 minutes"
                    exercises = "Yoga, Resistance Bands, Light Dumbbells"
                else:  # Diabetes
                    workout_type = "Mobility & Gentle Cardio"
                    frequency = "5 times a week"
                    duration = "35 minutes"
                    exercises = "Chair Yoga, Stretching, Walking"

    

    return workout_type, frequency, duration, exercises


def fetch_and_generate_exercise_plan(user_id, age, height, weight, gender):
    """Fetches user data from the database and generates an exercise plan"""

    with sqlite3.connect("predict.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT prediction FROM predictions WHERE user_id = ?", (user_id,))
        prediction_data = cursor.fetchone()

    prediction_result = prediction_data[0] if prediction_data else None

    # Calculate BMI and categorize it
    bmi = calculate_bmi(height, weight)
    bmi_category = categorize_bmi(bmi)

    # Generate exercise plan using user input + fetched prediction
    return get_exercise_plan(user_id,age, bmi_category, gender, prediction_result)

@app.route('/get_exercise_plan', methods=['POST'])
def get_exercise_plan_route():
    data = request.get_json()
    user_id = session.get("user_id")
    
    age = data.get('age')
    gender = data.get('gender')
    height = data.get('height')
    weight = data.get('weight')

    # Debugging print statements
    #print(f"📌 Received data: Age={age}, Gender={gender}, Height={height}, Weight={weight}")

    # Ensure all fields are present
    if not all([age, gender, height, weight]):
        print("❌ Missing data in request.")
        return jsonify({'error': 'Please provide all required details.'}), 400

    try:
        # Convert height and weight to float
        age = int(age)  
        height = float(height)
        weight = float(weight)

        #print(f"✅ Converted Data: Age={age} (type: {type(age)}), Height={height} (type: {type(height)}), Weight={weight} (type: {type(weight)})")

        # ✅ Calculate BMI
        bmi = calculate_bmi(height, weight)
        #print(f"🔍 Calculated BMI: {bmi} (type: {type(bmi)})")

        # ✅ Categorize BMI
        bmi_category = categorize_bmi(bmi)
        #print(f"📌 BMI Category: {bmi_category} (type: {type(bmi_category)})")

        prediction_result = None
        if user_id:
            with sqlite3.connect("predict.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT prediction FROM predictions WHERE user_id = ?", (user_id,))
                prediction_data = cursor.fetchone()
                prediction_result = prediction_data[0] if prediction_data else None

        #print(f"🧠 Prediction Result: {prediction_result}")

        # ✅ Ensure `prediction_result` is always passed
        if prediction_result is None:
            return jsonify({'error': 'Please complete your prediction first.'}), 400

        # ✅ Fetch exercise plan
        exercise_plan = get_exercise_plan(user_id,age, bmi_category, gender, prediction_result)

        # Store the generated exercise plan in database
        if exercise_plan:
            workout_type, frequency, duration, exercises = exercise_plan  # Unpack values
            
            # ✅ Store the structured exercise plan in database
            save_exercise_plan(user_id, workout_type, frequency, duration, exercises)
            print(f"✅ Exercise Plan Saved: {exercise_plan}")
            
        if exercise_plan:
            print(f"✅ Exercise Plan Found: {exercise_plan}")
            # ✅ Ensure correct JSON format
            return jsonify({
                "workout_type": exercise_plan[0],
                "frequency": exercise_plan[1],
                "duration": exercise_plan[2],
                "exercises": exercise_plan[3]
            })
        else:
            print("❌ No suitable exercise plan found in database.")
            return jsonify({'error': 'No suitable exercise plan found.'}), 404


    except ValueError as e:
        print(f"❌ ValueError: {e}")
        return jsonify({'error': 'Invalid height or weight value.'}), 400
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return jsonify({'error': str(e)}), 500
    
from datetime import datetime  
def save_exercise_plan(user_id, workout_type, frequency, duration, exercises):
    with sqlite3.connect("exercise.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO exercise (user_id, workout_type, frequency, duration, exercises, timestamp) 
                          VALUES (?, ?, ?, ?, ?, ?)''',
                       (user_id, workout_type, frequency, duration, exercises, datetime.now()))
        conn.commit()
 
 
@app.route('/meal', methods=['GET', 'POST'], endpoint='meal_page')
def meal():
    if 'user_id' not in session:
        flash('Please log in first.', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    prediction_result = get_prediction_result(user_id)

    if not prediction_result:
        flash("No prediction found. Please complete the prediction first.", "danger")
        return redirect(url_for('predict'))

    if request.method == 'POST':
        age = request.form.get('age')
        gender = request.form.get('gender')
        height = request.form.get('height')
        weight = request.form.get('weight')
        diet_preference = request.form.get('diet_preference')

        if not all([age, gender, height, weight, diet_preference]):
            flash("Please enter all required details.", "danger")
            return render_template('meal.html', age=age, gender=gender, height=height, weight=weight, diet_preference=diet_preference)

        

        try:
            age = int(age)
            height = float(height)
            weight = float(weight)

            bmi = calculate_bmi(height, weight)
            bmi_category = categorize_bmi(bmi)
            #age_group = get_age_group(age)
            
            # 4️⃣ Determine Age Group
            if age < 13:
                age_group = "Children"
            elif 13 <= age < 19:
                age_group = "Teens"
            elif 19 <= age < 60:
                age_group = "Adults"
            else:
                age_group = "Seniors"
                
            print(f"✅ Age Group: {age_group}, BMI Category: {bmi_category}")    

            # ✅ Fetch meal plan
            meal_plan = get_meal_recommendation(age_group, bmi_category, gender, diet_preference, prediction_result)

            if not meal_plan:
                flash("No suitable meal plan found.", "danger")
                return render_template('meal.html', age=age, gender=gender, height=height, weight=weight, diet_preference=diet_preference)

            return render_template('meal.html', meal_plan=meal_plan, age=age, gender=gender, height=height, weight=weight, diet_preference=diet_preference)

        except ValueError:
            flash("Invalid input. Please enter correct values.", "danger")
            return render_template('meal.html', age=age, gender=gender, height=height, weight=weight, diet_preference=diet_preference)

    return render_template('meal.html')
       
        
def fetch_and_generate_meal_plan(user_id, age, height, weight, gender, diet_preference):
    """Fetches prediction result, calculates BMI, determines age group & BMI category,
       generates a meal plan, and stores it in the database."""

    # 1️⃣ Fetch the latest prediction result from predict.db
    with sqlite3.connect("predict.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT prediction FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
        prediction_row = cursor.fetchone()
    
    prediction_result = prediction_row[0] if prediction_row else "No Prediction"

    # 2️⃣ Calculate BMI
    bmi = round(weight / ((height / 100) ** 2), 1)

    # 3️⃣ Determine BMI Category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # 4️⃣ Determine Age Group
    if age < 13:
        age_group = "Children"
    elif 13 <= age < 19:
        age_group = "Teens"
    elif 19 <= age < 60:
        age_group = "Adults"
    else:
        age_group = "Seniors"
        
    print(f"🔹 Age Group: {age_group}, BMI Category: {bmi_category}, Prediction: {prediction_result}")    

    # 5️⃣ Generate the meal plan for 7 days
    weekly_meals = get_meal_recommendation(age_group, bmi_category, gender, diet_preference, prediction_result)

    # 🔍 Debugging: Print meal plan structure
    print(f"DEBUG - Meal Plan Structure: {weekly_meals}")

    # 🛑 Check if the meal plan is empty or incorrectly formatted
    if not weekly_meals or not isinstance(weekly_meals, list) or len(weekly_meals) != 7:
        print("❌ No suitable meal plan found.")
        return "No suitable meal plan found."

    for i, meals in enumerate(weekly_meals):
        if not isinstance(meals, list) or len(meals) != 3:
            print(f"❌ Invalid meal structure on Day {i+1}: {meals}")
            return "No suitable meal plan found."

    print(f"✅ Generated Meal Plan: {weekly_meals}")

    # 6️⃣ Store the meal plan in meal.db
    with sqlite3.connect("meal.db") as conn:
        cursor = conn.cursor()

        for i, meals in enumerate(weekly_meals, start=1):  # ✅ Start from 1
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i-1]
            cursor.execute('''
                INSERT INTO meal (user_id, day, breakfast, lunch, dinner)
    VALUES (?, ?, ?, ?, ?)
            ''', (user_id, day_name, meals[0], meals[1], meals[2]))

        conn.commit()

    print("✅ 7-day meal plan generated and saved successfully!")

    return weekly_meals if weekly_meals else "No suitable meal plan found."       


from datetime import datetime

@app.route('/get_meal_plan', methods=['POST'])
def get_meal_plan():
    try:
        data = request.get_json()
        user_id = session.get("user_id")

        age = data.get('age')
        gender = data.get('gender')
        height = data.get('height')
        weight = data.get('weight')
        diet_preference = data.get('diet_preference')

        # ✅ Ensure all required fields are provided
        if not all([age, gender, height, weight, diet_preference]):
            print("❌ Missing data in request.")
            return jsonify({'error': 'Please provide all required details.'}), 400

        # ✅ Convert values to proper data types
        age = int(age)
        height = float(height)
        weight = float(weight)

        # ✅ Calculate BMI & categorize
        bmi = calculate_bmi(height, weight)
        bmi_category = categorize_bmi(bmi)
        #age_group = get_age_group(age)
        
        # ✅ Determine Age Group (Fixing missing function issue)
        if age < 13:
            age_group = "Children"
        elif 13 <= age < 19:
            age_group = "Teens"
        elif 19 <= age < 60:
            age_group = "Adults"
        else:
            age_group = "Seniors"

        # ✅ Fetch latest prediction result
        prediction_result = None
        if user_id:
            with sqlite3.connect("predict.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT prediction FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
                prediction_data = cursor.fetchone()
                prediction_result = prediction_data[0] if prediction_data else None

        # ✅ Ensure prediction result exists
        if not prediction_result:
            return jsonify({'error': 'Please complete your prediction first.'}), 400

        # ✅ Fetch meal plan based on parameters
        meal_plan = get_meal_recommendation(age_group, bmi_category, gender, diet_preference, prediction_result)

        # ✅ If no meal plan found, return error
        if not meal_plan:
            print("❌ No suitable meal plan found in database.")
            return jsonify({'error': 'No suitable meal plan found.'}), 404

        # ✅ Save meal plan in the database
        save_meal_plan(user_id, meal_plan)

        print(f"✅ Meal Plan Generated & Saved: {meal_plan}")

        # ✅ Ensure correct JSON response
        return jsonify({"meal_plan": meal_plan})

    except ValueError:
        return jsonify({"error": "Invalid height or weight value."}), 400
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return jsonify({"error": str(e)}), 500


# ✅ Function to store the meal plan in the database
def save_meal_plan(user_id, meal_plan):
    with sqlite3.connect("meal.db") as conn:
        cursor = conn.cursor()

        # ✅ Ensure meal plan format before saving
        for i, meals in enumerate(meal_plan, start=1):  # Start from 1
            if len(meals) != 3:
                print(f"❌ Skipping invalid meal entry for Day {i}: {meals}")
                continue  # Skip invalid entries

            day_label = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i-1]
            
            cursor.execute('''
                INSERT INTO meal (user_id, day, breakfast, lunch, dinner, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, day_label, meals[0], meals[1], meals[2], datetime.now()))
        conn.commit()

       
        
        
# Define possible meals based on user details
def get_meal_recommendation(age_group, bmi_category, gender, diet, prediction_result):
    
    


    # Ensure correct handling of age groups
    if age_group not in ["Children", "Teens", "Adults", "Seniors"]:
        print("❌ Invalid age group received!")
        return None

    print(age_group, bmi_category, gender, diet, prediction_result)
    
    weekly_meals  = []
    
    if age_group == "Children":
        if bmi_category == "Normal":
            if gender == "Male":
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Idli with coconut chutney", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Upma with peanuts", "Paneer paratha with curd", "Vegetable soup with chapati"],
                            ["Dhokla with mint chutney", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Besan cheela", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    else:  # Diabetes
                        weekly_meals = [
                            ["Ragi dosa", "Karela sabzi with chapati", "Lentil dal with brown rice"],
                            ["Methi thepla", "Lauki dal with quinoa", "Palak sabzi with roti"],
                            ["Vegetable upma", "Moong dal khichdi", "Bitter gourd curry with chapati"],
                            ["Sprouts salad", "Pumpkin curry with roti", "Vegetable stew with dal"],
                            ["Chia seed pudding", "Dal fry with rice", "Stuffed brinjal with roti"],
                            ["Oats porridge", "Chickpea salad with curd", "Ghiya curry with roti"],
                            ["Nuts and seeds mix", "Tofu sabzi with quinoa", "Rajma with brown rice"]
                        ]
                elif diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    else:  # Diabetes
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                elif diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Soy milk with poha", "Vegetable quinoa pulao", "Lentil soup with multigrain roti"],
                            ["Chia seed pudding", "Tofu stir-fry with brown rice", "Stuffed bell peppers with millet"],
                            ["Oats with almond milk", "Chickpea salad with tahini dressing", "Mushroom curry with rice"],
                            ["Sprouts chaat", "Lentil dal with quinoa", "Baked sweet potatoes with hummus"],
                            ["Peanut butter toast", "Stir-fried tofu with vegetables", "Vegetable stew with chapati"],
                            ["Banana smoothie with flaxseeds", "Grilled tofu sandwich", "Spinach lentil curry with rice"],
                            ["Coconut yogurt with fruits", "Vegan biryani with raita", "Sautéed vegetables with tempeh"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats porridge with flaxseeds", "Quinoa salad with tofu", "Vegetable stir-fry with brown rice"],
                            ["Almond milk smoothie", "Lentil soup with whole grain bread", "Stuffed zucchini with quinoa"],
                            ["Chickpea hummus with multigrain toast", "Tofu bhurji with roti", "Spinach and lentil dal"],
                            ["Sprouted moong salad", "Brown rice with vegetable curry", "Baked sweet potatoes with herbs"],
                            ["Chia seed pudding with nuts", "Bajra roti with dal", "Sautéed mushrooms and spinach"],
                            ["Peanut butter and banana toast", "Mixed vegetable quinoa khichdi", "Grilled tempeh with stir-fried veggies"],
                            ["Coconut yogurt with berries", "Vegan paneer curry with chapati", "Vegetable stew with red rice"]
                        ]
                    else:  # Diabetes
                        weekly_meals = [
                            ["Ragi porridge with nuts", "Quinoa and lentil salad", "Vegetable soup with millet bread"],
                            ["Flaxseed smoothie", "Stir-fried greens with tofu", "Stuffed bell peppers with dal"],
                            ["Oats upma with chia seeds", "Brown rice with sautéed vegetables", "Bitter gourd curry with roti"],
                            ["Sprouts and avocado salad", "Dal palak with chapati", "Baked pumpkin with herbs"],
                            ["Coconut yogurt with flaxseeds", "Besan chilla with mint chutney", "Tofu and stir-fried spinach"],
                            ["Almond butter toast", "Vegetable dalia khichdi", "Stuffed mushrooms with quinoa"],
                            ["Chia seed pudding", "Lauki dal with millet roti", "Sautéed okra with masoor dal"]
                        ]
                
                
    
            elif gender == "Female":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Soy milk with poha", "Vegetable quinoa pulao", "Lentil soup with multigrain roti"],
                            ["Chia seed pudding", "Tofu stir-fry with brown rice", "Stuffed bell peppers with millet"],
                            ["Oats with almond milk", "Chickpea salad with tahini dressing", "Mushroom curry with rice"],
                            ["Sprouts chaat", "Lentil dal with quinoa", "Baked sweet potatoes with hummus"],
                            ["Peanut butter toast", "Stir-fried tofu with vegetables", "Vegetable stew with chapati"],
                            ["Banana smoothie with flaxseeds", "Grilled tofu sandwich", "Spinach lentil curry with rice"],
                            ["Coconut yogurt with fruits", "Vegan biryani with raita", "Sautéed vegetables with tempeh"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats porridge with flaxseeds", "Quinoa salad with tofu", "Vegetable stir-fry with brown rice"],
                            ["Almond milk smoothie", "Lentil soup with whole grain bread", "Stuffed zucchini with quinoa"],
                            ["Chickpea hummus with multigrain toast", "Tofu bhurji with roti", "Spinach and lentil dal"],
                            ["Sprouted moong salad", "Brown rice with vegetable curry", "Baked sweet potatoes with herbs"],
                            ["Chia seed pudding with nuts", "Bajra roti with dal", "Sautéed mushrooms and spinach"],
                            ["Peanut butter and banana toast", "Mixed vegetable quinoa khichdi", "Grilled tempeh with stir-fried veggies"],
                            ["Coconut yogurt with berries", "Vegan paneer curry with chapati", "Vegetable stew with red rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi porridge with almonds", "Quinoa and lentil salad", "Vegetable soup with millet bread"],
                            ["Flaxseed smoothie", "Stir-fried greens with tofu", "Stuffed bell peppers with dal"],
                            ["Oats upma with chia seeds", "Brown rice with sautéed vegetables", "Bitter gourd curry with roti"],
                            ["Sprouts and avocado salad", "Dal palak with chapati", "Baked pumpkin with herbs"],
                            ["Coconut yogurt with flaxseeds", "Besan chilla with mint chutney", "Tofu and stir-fried spinach"],
                            ["Almond butter toast", "Vegetable dalia khichdi", "Stuffed mushrooms with quinoa"],
                            ["Chia seed pudding", "Lauki dal with millet roti", "Sautéed okra with masoor dal"]
                        ]
                        
                elif diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                            ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                        
                elif diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
                        
        elif bmi_category == "Overweight":
            if gender == "Female":
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Grilled chicken with vegetables", "Fish curry with brown rice", "Boiled eggs with chapati"],
                        ["Scrambled egg whites", "Steamed fish with quinoa", "Chicken salad"],
                        ["Egg bhurji with multigrain toast", "Tandoori fish with dal", "Grilled prawns with veggies"],
                        ["Oats with boiled eggs", "Chicken stir-fry with roti", "Baked salmon with sautéed greens"],
                        ["Egg white omelette", "Mutton stew with brown rice", "Prawn curry with chapati"],
                        ["Chia pudding with almonds", "Turkey curry with multigrain roti", "Steamed fish with vegetables"],
                        ["Peanut butter toast", "Egg salad with greens", "Grilled chicken with lentil soup"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with egg whites", "Grilled chicken with quinoa", "Steamed fish with vegetables"],
                        ["Boiled eggs with avocado", "Mutton soup with chapati", "Grilled prawns with salad"],
                        ["Scrambled egg whites", "Turkey stir-fry with rice", "Fish tikka with dal"],
                        ["Egg salad", "Grilled salmon with brown rice", "Chicken kebabs with veggies"],
                        ["Egg bhurji with oats", "Lentil soup with grilled fish", "Turkey curry with chapati"],
                        ["Scrambled egg wrap", "Baked fish with salad", "Chicken stew with roti"],
                        ["Poached eggs with spinach", "Steamed prawns with lentils", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed spinach", "Steamed fish with brown rice", "Chicken soup with roti"],
                        ["Boiled egg with whole wheat toast", "Grilled salmon with quinoa", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with methi", "Baked chicken with salad", "Prawn curry with chapati"],
                        ["Oats with egg whites", "Steamed fish with dal", "Grilled chicken with brown rice"],
                        ["Egg white omelette with veggies", "Turkey stir-fry with multigrain roti", "Grilled prawns with salad"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with sautéed greens"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
            
                elif diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Almond milk smoothie", "Quinoa salad with chickpeas", "Stir-fried tofu with brown rice"],
                        ["Chia pudding with nuts", "Lentil soup with multigrain bread", "Stuffed bell peppers with quinoa"],
                        ["Oats with flaxseeds", "Vegetable khichdi with curd", "Grilled mushrooms with roti"],
                        ["Banana smoothie with seeds", "Chickpea curry with rice", "Mixed vegetable curry with chapati"],
                        ["Tofu scramble with toast", "Rajma chawal", "Stuffed paratha with vegan yogurt"],
                        ["Besan cheela with mint chutney", "Palak dal with rice", "Vegan methi thepla with cucumber raita"],
                        ["Peanut butter toast", "Vegetable stew with chapati", "Baked sweet potatoes with spices"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Moong dal cheela with mint chutney", "Sautéed spinach with quinoa"],
                        ["Sprouts salad", "Lauki dal with roti", "Stuffed capsicum with tofu"],
                        ["Flaxseed smoothie", "Vegetable upma with coconut chutney", "Tofu stir-fry with brown rice"],
                        ["Banana and almond smoothie", "Chana masala with quinoa", "Vegetable curry with bajra roti"],
                        ["Chia pudding with seeds", "Methi paratha with plant-based curd", "Lauki sabzi with chapati"],
                        ["Oats porridge with walnuts", "Rajma with brown rice", "Masoor dal with roti"],
                        ["Besan pancake", "Vegetable pulao with cucumber raita", "Baked tofu with sautéed greens"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi porridge with nuts", "Bitter gourd stir-fry with roti", "Moong dal khichdi"],
                        ["Chia seed pudding with nuts", "Tofu curry with quinoa", "Lauki dal with chapati"],
                        ["Besan cheela with coriander chutney", "Brown rice with vegetable curry", "Grilled eggplant with roti"],
                        ["Sprouts chaat", "Vegetable daliya", "Methi thepla with coconut chutney"],
                        ["Flaxseed smoothie", "Karela sabzi with dal", "Bajra roti with mixed vegetable curry"],
                        ["Oats upma with seeds", "Lentil soup with multigrain bread", "Sautéed okra with tofu"],
                        ["Peanut butter toast", "Rajma with brown rice", "Masoor dal with tandoori roti"]
                    ]
            if gender == "Male":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Soy milk smoothie", "Quinoa stir-fry with vegetables", "Tofu curry with brown rice"],
                        ["Chia pudding with berries", "Lentil soup with whole wheat bread", "Stuffed bell peppers with quinoa"],
                        ["Oats with almond butter", "Vegetable khichdi", "Grilled tofu with roti"],
                        ["Banana smoothie with flaxseeds", "Chickpea curry with rice", "Mixed vegetable curry with chapati"],
                        ["Tofu scramble with toast", "Rajma chawal", "Vegan stuffed paratha with coconut yogurt"],
                        ["Besan cheela with chutney", "Palak dal with brown rice", "Methi thepla with plant-based curd"],
                        ["Peanut butter toast", "Vegetable stew with chapati", "Roasted sweet potatoes with greens"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with chia seeds", "Moong dal cheela with chutney", "Sauteed spinach with quinoa"],
                        ["Sprouts salad", "Lauki dal with roti", "Stuffed capsicum with tofu"],
                        ["Flaxseed smoothie", "Vegetable upma with coconut chutney", "Tofu stir-fry with brown rice"],
                        ["Banana almond shake", "Chana masala with quinoa", "Vegetable curry with bajra roti"],
                        ["Chia pudding with walnuts", "Methi paratha with plant-based curd", "Lauki sabzi with chapati"],
                        ["Oats porridge with sunflower seeds", "Rajma with brown rice", "Masoor dal with roti"],
                        ["Besan pancake", "Vegetable pulao with cucumber raita", "Baked tofu with sauteed greens"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi porridge with nuts", "Bitter gourd stir-fry with roti", "Moong dal khichdi"],
                        ["Chia seed pudding with nuts", "Tofu curry with quinoa", "Lauki dal with chapati"],
                        ["Besan cheela with coriander chutney", "Brown rice with vegetable curry", "Grilled eggplant with roti"],
                        ["Sprouts chaat", "Vegetable daliya", "Methi thepla with coconut chutney"],
                        ["Flaxseed smoothie", "Karela sabzi with dal", "Bajra roti with mixed vegetable curry"],
                        ["Oats upma with seeds", "Lentil soup with multigrain bread", "Sauteed okra with tofu"],
                        ["Peanut butter toast", "Rajma with brown rice", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
        elif bmi_category == "Underweight":
            if gender == "Male":
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Banana shake with almonds", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Milk with oats", "Rajma chawal", "Baingan bharta with roti"],
                        ["Sprouts salad", "Masoor dal with rice", "Methi thepla with yogurt"],
                        ["Chia pudding", "Palak paneer with chapati", "Pumpkin curry with chapati"],
                        ["Vegetable poha", "Dal khichdi with ghee", "Stuffed capsicum with roti"],
                        ["Moong dal cheela", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Sprouts chaat", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Methi paratha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Vegetable poha", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Banana shake with nuts", "Chana masala with roti", "Kadhi with brown rice"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Tofu curry with chapati", "Lauki sabzi with dal"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Boiled eggs with toast", "Chicken curry with rice", "Grilled fish with chapati"],
                        ["Omelette with whole wheat bread", "Fish stew with quinoa", "Tandoori chicken with salad"],
                        ["Scrambled eggs with spinach", "Mutton curry with jeera rice", "Grilled prawns with roti"],
                        ["Egg bhurji with paratha", "Butter chicken with rice", "Grilled salmon with vegetables"],
                        ["Egg dosa with chutney", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Boiled eggs with multigrain toast", "Chicken tikka with salad", "Mutton soup with whole wheat bread"],
                        ["Egg salad sandwich", "Fish masala with roti", "Chicken kebabs with quinoa"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Soy milk with granola", "Lentil soup with brown rice", "Grilled tofu with chapati"],
                        ["Chia pudding with nuts", "Vegetable stir-fry with quinoa", "Stuffed bell peppers"],
                        ["Peanut butter toast", "Rajma with jeera rice", "Sauteed spinach with tofu"],
                        ["Oatmeal with almond butter", "Mixed vegetable curry with chapati", "Pumpkin soup with whole wheat bread"],
                        ["Banana smoothie with flaxseeds", "Chickpea curry with rice", "Grilled mushrooms with dal"],
                        ["Besan cheela with mint chutney", "Vegetable biryani with raita", "Sautéed broccoli with quinoa"],
                        ["Tofu scramble with toast", "Dal khichdi with ghee", "Lauki sabzi with roti"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Chia seed smoothie", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Oats with nuts", "Lauki dal with chapati", "Palak tofu with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed mushrooms with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak tofu with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
    elif age_group == "Teens":
        if bmi_category == "Overweight":
            if gender == "Male":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Oats with flaxseeds", "Lentil soup with quinoa", "Grilled tofu with stir-fried veggies"],
                        ["Smoothie with almond milk", "Vegetable khichdi", "Chickpea salad with lemon dressing"],
                        ["Chia pudding with nuts", "Brown rice with dal", "Stuffed bell peppers"],
                        ["Besan cheela with mint chutney", "Mixed vegetable curry with chapati", "Pumpkin soup with whole wheat bread"],
                        ["Banana smoothie with flaxseeds", "Tofu stir-fry with quinoa", "Sautéed broccoli with dal"],
                        ["Sprouts salad", "Vegetable daliya", "Karela sabzi with roti"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia seed smoothie", "Lauki dal with chapati", "Palak tofu with quinoa"],
                        ["Oats upma", "Karela dal with rice", "Stuffed mushrooms with roti"],
                        ["Sprouts chaat", "Vegetable daliya", "Methi thepla with curd"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak tofu with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                elif diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
            elif gender == "Female":
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Almond milk smoothie", "Quinoa salad with hummus", "Lentil soup with whole wheat bread"],
                        ["Chia pudding with berries", "Grilled tofu with brown rice", "Vegetable stir-fry with quinoa"],
                        ["Oats porridge with nuts", "Chickpea curry with chapati", "Stuffed bell peppers with rice"],
                        ["Banana peanut butter toast", "Vegetable dalia", "Spinach and tofu stir-fry"],
                        ["Soy milk with granola", "Mushroom curry with rice", "Baked sweet potatoes with lentils"],
                        ["Vegan pancakes with maple syrup", "Rajma with quinoa", "Pumpkin soup with whole grain bread"],
                        ["Smoothie bowl with flaxseeds", "Vegetable khichdi", "Stuffed paratha with chutney"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats porridge with almonds", "Lentil salad with lemon dressing", "Quinoa vegetable pulao"],
                        ["Green smoothie with chia", "Tofu bhurji with whole wheat toast", "Methi thepla with curd"],
                        ["Chia seed pudding", "Vegetable stew with multigrain bread", "Lauki sabzi with dal"],
                        ["Fruit salad with seeds", "Chickpea chaat with roti", "Baked tofu with quinoa"],
                        ["Flaxseed smoothie", "Brown rice with moong dal", "Sautéed spinach with garlic"],
                        ["Multigrain toast with avocado", "Vegetable stir-fry with couscous", "Masoor dal with chapati"],
                        ["Soy yogurt with nuts", "Rajma with millet roti", "Lentil soup with whole wheat bread"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi porridge with nuts", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Flaxseed smoothie", "Lentil soup with quinoa", "Palak tofu with chapati"],
                        ["Sprouts salad with lemon", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Vegetable upma", "Tofu stir-fry with quinoa", "Methi thepla with curd"],
                        ["Oats idli with chutney", "Karela sabzi with roti", "Spinach and lentil curry"],
                        ["Soy yogurt with flaxseeds", "Rajma with brown rice", "Masoor dal with tandoori roti"],
                        ["Multigrain toast with almond butter", "Vegetable daliya", "Pumpkin soup with whole wheat bread"]
                    ]
        elif bmi_category == "Normal":
            if gender == "Female":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Oats porridge with almonds", "Vegetable quinoa salad", "Lentil soup with whole wheat bread"],
                        ["Chia pudding with berries", "Grilled tofu with brown rice", "Vegetable stir-fry with quinoa"],
                        ["Banana smoothie with flaxseeds", "Chickpea curry with chapati", "Stuffed bell peppers with rice"],
                        ["Peanut butter toast", "Vegetable dalia", "Spinach and tofu stir-fry"],
                        ["Soy milk with granola", "Mushroom curry with rice", "Baked sweet potatoes with lentils"],
                        ["Vegan pancakes with maple syrup", "Rajma with quinoa", "Pumpkin soup with whole grain bread"],
                        ["Smoothie bowl with chia seeds", "Vegetable khichdi", "Stuffed paratha with chutney"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats porridge with flaxseeds", "Lentil salad with lemon dressing", "Quinoa vegetable pulao"],
                        ["Green smoothie with chia", "Tofu bhurji with whole wheat toast", "Methi thepla with curd"],
                        ["Chia seed pudding", "Vegetable stew with multigrain bread", "Lauki sabzi with dal"],
                        ["Fruit salad with seeds", "Chickpea chaat with roti", "Baked tofu with quinoa"],
                        ["Flaxseed smoothie", "Brown rice with moong dal", "Sautéed spinach with garlic"],
                        ["Multigrain toast with avocado", "Vegetable stir-fry with couscous", "Masoor dal with chapati"],
                        ["Soy yogurt with nuts", "Rajma with millet roti", "Lentil soup with whole wheat bread"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi porridge with nuts", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Flaxseed smoothie", "Lentil soup with quinoa", "Palak tofu with chapati"],
                        ["Sprouts salad with lemon", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Vegetable upma", "Tofu stir-fry with quinoa", "Methi thepla with curd"],
                        ["Oats idli with chutney", "Karela sabzi with roti", "Spinach and lentil curry"],
                        ["Soy yogurt with flaxseeds", "Rajma with brown rice", "Masoor dal with tandoori roti"],
                        ["Multigrain toast with almond butter", "Vegetable daliya", "Pumpkin soup with whole wheat bread"]
                    ]
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
            if gender == "Male":
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Almond milk with chia seeds", "Quinoa salad with chickpeas", "Lentil soup with whole wheat bread"],
                        ["Tofu scramble with veggies", "Brown rice with stir-fried tofu", "Vegetable stew with quinoa"],
                        ["Oats with flaxseeds", "Sweet potato curry with chapati", "Stuffed bell peppers with lentils"],
                        ["Peanut butter toast", "Vegan rajma chawal", "Stir-fried mixed vegetables with tofu"],
                        ["Banana smoothie with almond butter", "Bajra roti with vegetable curry", "Spinach dal with rice"],
                        ["Chickpea pancakes", "Pumpkin soup with bread", "Quinoa pulao with salad"],
                        ["Soy milk smoothie", "Vegetable biryani with raita", "Mushroom curry with roti"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with walnuts", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Tofu stir-fry with brown rice", "Spinach curry with chapati"],
                        ["Methi paratha", "Rajma with quinoa", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with multigrain bread", "Bitter gourd curry with dal"],
                        ["Banana shake with nuts", "Stir-fried broccoli with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with almond curd", "Dal fry with millet roti"],
                        ["Cucumber avocado sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with coconut chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with flaxseeds", "Lauki dal with chapati", "Spinach curry with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
        if bmi_category == "Underweight":
            if gender == "Male":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Almond butter toast", "Quinoa with lentils", "Vegetable soup with tofu"],
                        ["Oats with flaxseeds", "Brown rice with stir-fried vegetables", "Chickpea curry with roti"],
                        ["Banana smoothie with nuts", "Vegetable dalia", "Mushroom stir-fry with quinoa"],
                        ["Peanut butter toast", "Vegan rajma chawal", "Spinach lentil soup with whole wheat bread"],
                        ["Soy milk smoothie", "Bajra roti with mixed vegetables", "Lentil stew with quinoa"],
                        ["Chickpea pancakes", "Sweet potato curry with rice", "Vegetable biryani with almond yogurt"],
                        ["Tofu scramble with avocado", "Millet khichdi with ghee", "Sautéed vegetables with roti"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Bitter gourd stir-fry with brown rice"],
                        ["Sprouts salad", "Vegetable stew with millet roti", "Tofu curry with quinoa"],
                        ["Besan cheela with mint chutney", "Dal khichdi with ghee", "Stuffed bell peppers with lentils"],
                        ["Vegetable poha", "Rajma with quinoa", "Spinach curry with roti"],
                        ["Flaxseed smoothie", "Tofu stir-fry with multigrain bread", "Okra sautéed with dal"],
                        ["Methi paratha", "Chana masala with brown rice", "Karela sabzi with roti"],
                        ["Oats upma", "Lentil soup with quinoa", "Palak paneer with millet roti"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with coconut chutney", "Moong dal khichdi", "Bajra roti with karela sabzi"],
                        ["Chia pudding with flaxseeds", "Lauki dal with chapati", "Spinach lentil soup with quinoa"],
                        ["Besan cheela with tomato chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts chaat", "Vegetable daliya", "Methi thepla with almond curd"],
                        ["Oats with nuts", "Karela dal with rice", "Lauki curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with quinoa", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
            if gender == "Female":
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                        ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                        ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                        ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                        ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                        ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                        ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                        ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                        ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                        ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                        ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                        ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                        ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                        ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                        ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                        ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                        ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                        ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                        ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                    ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                        ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                        ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                        ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                        ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                        ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                        ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                        ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                        ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                        ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                        ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                        ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                        ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                        ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                        ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                        ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                        ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                        ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                        ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                    ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                        ["Almond milk smoothie", "Chickpea salad", "Vegetable stir-fry with quinoa"],
                        ["Tofu scramble with toast", "Lentil soup with whole grain bread", "Stuffed bell peppers"],
                        ["Peanut butter banana toast", "Vegan biryani with raita", "Grilled tofu with sautéed spinach"],
                        ["Oats porridge with nuts", "Quinoa salad with chickpeas", "Lentil stew with rice"],
                        ["Chia pudding with berries", "Vegetable curry with millet", "Tofu stir-fry with brown rice"],
                        ["Smoothie bowl with seeds", "Pumpkin soup with bread", "Mushroom stir-fry with quinoa"],
                        ["Whole wheat avocado toast", "Hummus with vegetable wrap", "Stir-fried tempeh with rice"]
                    ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                        ["Green smoothie", "Sprouted moong salad", "Steamed vegetables with quinoa"],
                        ["Oats with flaxseeds", "Vegetable dalia", "Tofu stir-fry with greens"],
                        ["Chia pudding", "Bajra roti with mixed vegetables", "Lauki sabzi with dal"],
                        ["Sprouts with lemon", "Quinoa upma", "Stuffed paratha with curd"],
                        ["Flaxseed smoothie", "Methi thepla with dal", "Vegetable stew with brown rice"],
                        ["Nut butter toast", "Vegetable khichdi", "Cabbage stir-fry with tofu"],
                        ["Avocado smoothie", "Dal fry with jowar roti", "Vegetable pulao with coconut chutney"]
                    ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                        ["Oats with cinnamon", "Lentil soup with quinoa", "Sauteed greens with tofu"],
                        ["Flaxseed porridge", "Sprouted lentil salad", "Mixed vegetable curry with bajra roti"],
                        ["Green smoothie with chia", "Ragi dosa with chutney", "Pumpkin curry with rice"],
                        ["Almond butter toast", "Vegetable dalia", "Stuffed brinjal with roti"],
                        ["Sprouts chaat", "Spinach dal with brown rice", "Lauki sabzi with chapati"],
                        ["Chia pudding with nuts", "Jowar roti with dal", "Karela stir-fry with lentils"],
                        ["Coconut smoothie", "Moong dal chilla", "Vegetable curry with foxtail millet"]
                    ]
            
    elif age_group == "Adults":
        if bmi_category == "Overweight":
            if gender == "Female":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Almond milk smoothie", "Chickpea salad", "Vegetable stir-fry with quinoa"],
                            ["Tofu scramble with toast", "Lentil soup with whole grain bread", "Stuffed bell peppers"],
                            ["Peanut butter banana toast", "Vegan biryani with raita", "Grilled tofu with sautéed spinach"],
                            ["Oats porridge with nuts", "Quinoa salad with chickpeas", "Lentil stew with rice"],
                            ["Chia pudding with berries", "Vegetable curry with millet", "Tofu stir-fry with brown rice"],
                            ["Smoothie bowl with seeds", "Pumpkin soup with bread", "Mushroom stir-fry with quinoa"],
                            ["Whole wheat avocado toast", "Hummus with vegetable wrap", "Stir-fried tempeh with rice"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Green smoothie", "Sprouted moong salad", "Steamed vegetables with quinoa"],
                            ["Oats with flaxseeds", "Vegetable dalia", "Tofu stir-fry with greens"],
                            ["Chia pudding", "Bajra roti with mixed vegetables", "Lauki sabzi with dal"],
                            ["Sprouts with lemon", "Quinoa upma", "Stuffed paratha with curd"],
                            ["Flaxseed smoothie", "Methi thepla with dal", "Vegetable stew with brown rice"],
                            ["Nut butter toast", "Vegetable khichdi", "Cabbage stir-fry with tofu"],
                            ["Avocado smoothie", "Dal fry with jowar roti", "Vegetable pulao with coconut chutney"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Oats with cinnamon", "Lentil soup with quinoa", "Sauteed greens with tofu"],
                            ["Flaxseed porridge", "Sprouted lentil salad", "Mixed vegetable curry with bajra roti"],
                            ["Green smoothie with chia", "Ragi dosa with chutney", "Pumpkin curry with rice"],
                            ["Almond butter toast", "Vegetable dalia", "Stuffed brinjal with roti"],
                            ["Sprouts chaat", "Spinach dal with brown rice", "Lauki sabzi with chapati"],
                            ["Chia pudding with nuts", "Jowar roti with dal", "Karela stir-fry with lentils"],
                            ["Coconut smoothie", "Moong dal chilla", "Vegetable curry with foxtail millet"]
                        ]
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                            ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
            if gender == "Male":
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                            ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Almond milk smoothie", "Quinoa salad with chickpeas", "Lentil soup with whole grain bread"],
                            ["Chia pudding with berries", "Stir-fried tofu with vegetables", "Grilled mushrooms with quinoa"],
                            ["Oats with flaxseeds", "Brown rice with mixed beans", "Sweet potato curry with millet"],
                            ["Peanut butter toast", "Lentil stew with whole wheat bread", "Stuffed bell peppers"],
                            ["Green smoothie with spinach", "Tofu scramble with veggies", "Pumpkin soup with quinoa"],
                            ["Fruit salad with nuts", "Chickpea curry with brown rice", "Vegetable stir-fry with millet"],
                            ["Soy yogurt with seeds", "Whole wheat pasta with tomato sauce", "Baked sweet potatoes with tahini"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with walnuts", "Quinoa bowl with avocado", "Vegetable soup with lentils"],
                            ["Flaxseed smoothie", "Stir-fried tofu with brown rice", "Baked zucchini with hummus"],
                            ["Chia seed pudding", "Buckwheat salad with nuts", "Stuffed eggplant with chickpeas"],
                            ["Whole grain toast with almond butter", "Lentil soup with vegetables", "Grilled mushrooms with quinoa"],
                            ["Berry smoothie with chia", "Vegetable stir-fry with tofu", "Spinach and lentil dal with brown rice"],
                            ["Banana with peanut butter", "Chickpea stew with quinoa", "Roasted vegetables with millet"],
                            ["Soy yogurt with flaxseeds", "Kale and tofu stir-fry", "Vegetable curry with whole wheat bread"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Chia pudding with almonds", "Steamed vegetables with lentils", "Grilled tofu with stir-fried greens"],
                            ["Oats with flaxseeds", "Buckwheat salad with avocado", "Baked sweet potatoes with tahini"],
                            ["Nut-based smoothie", "Lentil soup with spinach", "Stir-fried zucchini with tofu"],
                            ["Whole grain toast with peanut butter", "Vegetable curry with quinoa", "Grilled eggplant with hummus"],
                            ["Berry and almond smoothie", "Chickpea and kale stir-fry", "Mushroom soup with whole wheat bread"],
                            ["Soy yogurt with seeds", "Brown rice with vegetable stew", "Spinach and lentil dal"],
                            ["Green smoothie with walnuts", "Baked vegetables with quinoa", "Tofu scramble with roasted peppers"]
                        ]
                        
        if bmi_category == "Underweight":
            if gender == "Male":
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Peanut butter banana smoothie", "Chickpea curry with rice", "Lentil soup with whole wheat bread"],
                            ["Chia seed pudding with nuts", "Quinoa and tofu stir-fry", "Grilled sweet potatoes with tahini"],
                            ["Almond milk oatmeal with dates", "Vegetable dal with roti", "Stuffed bell peppers with lentils"],
                            ["Soy yogurt with berries", "Brown rice with mixed beans", "Tofu scramble with vegetables"],
                            ["Flaxseed smoothie", "Lentil stew with quinoa", "Roasted vegetables with hummus"],
                            ["Nut-based smoothie", "Vegetable stir-fry with chickpeas", "Buckwheat salad with avocado"],
                            ["Green smoothie with walnuts", "Baked tofu with spinach", "Grilled eggplant with whole wheat bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almond butter", "Quinoa and spinach stir-fry", "Grilled zucchini with lentils"],
                            ["Chia seed pudding with walnuts", "Buckwheat and vegetable soup", "Mushroom stir-fry with tofu"],
                            ["Soy milk smoothie", "Lentil dal with quinoa", "Roasted pumpkin with tahini"],
                            ["Whole grain toast with peanut butter", "Chickpea curry with brown rice", "Spinach and lentil stew"],
                            ["Berry and flaxseed smoothie", "Vegetable stir-fry with tofu", "Baked sweet potatoes with hummus"],
                            ["Almond milk with chia seeds", "Brown rice and mixed vegetables", "Grilled mushrooms with avocado"],
                            ["Soy yogurt with nuts", "Lentil soup with spinach", "Tofu and kale stir-fry"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Chia pudding with almonds", "Steamed vegetables with lentils", "Grilled tofu with stir-fried greens"],
                            ["Oats with flaxseeds", "Buckwheat salad with avocado", "Baked sweet potatoes with tahini"],
                            ["Nut-based smoothie", "Lentil soup with spinach", "Stir-fried zucchini with tofu"],
                            ["Whole grain toast with peanut butter", "Vegetable curry with quinoa", "Grilled eggplant with hummus"],
                            ["Berry and almond smoothie", "Chickpea and kale stir-fry", "Mushroom soup with whole wheat bread"],
                            ["Soy yogurt with seeds", "Brown rice with vegetable stew", "Spinach and lentil dal"],
                            ["Green smoothie with walnuts", "Baked vegetables with quinoa", "Tofu scramble with roasted peppers"]
                        ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                            ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
            if gender == "Female":
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                            ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                        
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Oats with almond milk", "Quinoa salad with chickpeas", "Vegetable stir-fry with tofu"],
                            ["Chia seed pudding", "Lentil soup with whole wheat bread", "Stuffed bell peppers"],
                            ["Banana smoothie with flaxseeds", "Brown rice with mixed vegetables", "Grilled mushroom skewers"],
                            ["Peanut butter on multigrain toast", "Sweet potato curry with rice", "Vegan stuffed cabbage rolls"],
                            ["Soy milk porridge", "Tofu stir-fry with quinoa", "Dal with millet roti"],
                            ["Vegan pancakes with berries", "Kale and chickpea stew", "Baked eggplant with tomato sauce"],
                            ["Green smoothie with nuts", "Vegetable biryani with raita", "Lentil-stuffed zucchini"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Quinoa porridge with chia", "Vegetable soup with lentils", "Baked tofu with sautéed greens"],
                            ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                            ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                            ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                            ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                            ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                            ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                            ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                            ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                            ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                            ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                            ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                            ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                        ]
                        
                        
    if bmi_category == "Normal":
        if gender == "Female":
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                     weekly_meals = [
                    ["Green smoothie with chia", "Quinoa and black bean salad", "Grilled tofu with vegetables"],
                    ["Overnight oats with almond milk", "Lentil soup with whole wheat bread", "Stuffed bell peppers"],
                    ["Fruit and nut parfait", "Steamed broccoli with chickpeas", "Vegetable stir-fry with tofu"],
                    ["Chia pudding with flaxseeds", "Brown rice with mixed vegetables", "Baked sweet potato with tahini"],
                    ["Vegan pancakes with berries", "Tofu stir-fry with quinoa", "Dal with millet roti"],
                    ["Soy yogurt with nuts", "Vegetable biryani with raita", "Lentil-stuffed zucchini"],
                    ["Banana smoothie with seeds", "Chickpea and kale stew", "Grilled eggplant with tahini"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almond milk", "Vegetable soup with lentils", "Grilled tempeh with sautéed greens"],
                    ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                    ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                    ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                    ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                    ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                    ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                    ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                    ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                    ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                    ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                    ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                    ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                ]
                        
            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Vegetable upma", "Palak paneer with chapati", "Dal with brown rice"],
                    ["Besan cheela with mint chutney", "Vegetable pulao with curd", "Stuffed capsicum with roti"],
                    ["Banana smoothie with seeds", "Chickpea salad with lemon dressing", "Methi thepla with yogurt"],
                    ["Oats porridge with flaxseeds", "Dal khichdi with ghee", "Rajma with whole wheat roti"],
                    ["Idli with sambhar", "Vegetable stir-fry with tofu", "Lauki sabzi with chapati"],
                    ["Moong dal dosa with chutney", "Bhindi masala with brown rice", "Baingan bharta with roti"],
                    ["Sprouts chaat", "Curd rice with pickle", "Pumpkin curry with chapati"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                    ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                    ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                    ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                    ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                    ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                    ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                    ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                    ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                    ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                    ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                    ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                    ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                ]
                        
            if gender == "Female":
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
            if gender == "Male":
                if diet == "Non-Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Egg dosa with chutney", "Chicken curry with rice", "Grilled fish with roti"],
                            ["Omelette with whole wheat toast", "Fish curry with brown rice", "Chicken stew with chapati"],
                            ["Scrambled eggs with vegetables", "Mutton curry with jeera rice", "Tandoori chicken with naan"],
                            ["Boiled eggs with bread", "Egg biryani with raita", "Prawn masala with roti"],
                            ["Egg bhurji with paratha", "Chicken tikka with rice", "Grilled salmon with sautéed veggies"],
                            ["Egg paratha", "Keema curry with chapati", "Fish fry with dal and rice"],
                            ["Egg salad sandwich", "Butter chicken with rice", "Mutton soup with bread"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Egg white scramble", "Grilled chicken with brown rice", "Fish stew with vegetables"],
                            ["Boiled eggs with toast", "Tandoori fish with quinoa", "Chicken curry with chapati"],
                            ["Egg omelette with spinach", "Mutton stew with rice", "Grilled prawns with veggies"],
                            ["Egg salad", "Chicken stir-fry with multigrain roti", "Grilled salmon with quinoa"],
                            ["Egg bhurji with oats", "Turkey curry with rice", "Fish tikka with dal"],
                            ["Scrambled egg wrap", "Egg curry with chapati", "Chicken kebabs with salad"],
                            ["Poached eggs with avocado", "Fish masala with roti", "Grilled chicken with vegetables"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Egg whites with sautéed veggies", "Steamed fish with brown rice", "Chicken soup with chapati"],
                            ["Boiled egg with multigrain toast", "Grilled salmon with salad", "Turkey stew with vegetables"],
                            ["Scrambled egg whites with spinach", "Baked chicken with quinoa", "Prawn curry with roti"],
                            ["Oats with egg whites", "Steamed fish with lentils", "Grilled chicken with brown rice"],
                            ["Egg white omelette", "Turkey stir-fry with chapati", "Grilled prawns with vegetables"],
                            ["Egg bhurji with multigrain bread", "Lentil soup with grilled fish", "Chicken tikka with salad"],
                            ["Egg white wrap", "Baked fish with vegetables", "Mutton soup with whole wheat bread"]
                        ]
                        
                if diet == "Veg":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Milk with poha", "Vegetable khichdi with curd", "Dal with rice and ghee"],
                            ["Chia pudding with nuts", "Paneer paratha with curd", "Vegetable stew with chapati"],
                            ["Besan cheela", "Lentil dal with jeera rice", "Stuffed capsicum with roti"],
                            ["Aloo poha", "Rajma chawal", "Mixed vegetable curry with roti"],
                            ["Moong dal cheela", "Masoor dal with brown rice", "Baingan bharta with roti"],
                            ["Vegetable uttapam", "Palak paneer with chapati", "Methi thepla with yogurt"],
                            ["Banana smoothie with seeds", "Curd rice with pickle", "Pumpkin curry with chapati"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                            ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                            ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                            ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                            ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                            ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                            ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                        
                if diet == "Vegan":
                    if prediction_result == "Non-Diabetic":
                        weekly_meals = [
                            ["Smoothie with almond milk", "Quinoa salad with chickpeas", "Lentil soup with whole wheat bread"],
                            ["Oats with nuts", "Grilled tofu with brown rice", "Stuffed bell peppers"],
                            ["Besan cheela with mint chutney", "Vegetable stir-fry with quinoa", "Chickpea curry with roti"],
                            ["Sprouts salad", "Vegan biryani with raita", "Stuffed zucchini boats"],
                            ["Chia pudding with flaxseeds", "Tofu curry with roti", "Pumpkin soup with quinoa"],
                            ["Peanut butter toast", "Vegan dal makhani with brown rice", "Sautéed spinach with lentils"],
                            ["Banana oat pancakes", "Vegetable curry with millet", "Mushroom stir-fry with rice"]
                        ]
                    elif prediction_result == "Pre-Diabetic":
                        weekly_meals = [
                            ["Chia seed pudding", "Quinoa and vegetable pilaf", "Lentil and spinach stew"],
                            ["Oats with almond butter", "Tofu stir-fry with brown rice", "Stuffed capsicum with chickpeas"],
                            ["Green smoothie", "Chickpea salad with tahini dressing", "Vegan mixed dal with roti"],
                            ["Sprouted moong chaat", "Vegetable and lentil khichdi", "Grilled mushrooms with quinoa"],
                            ["Flaxseed pancakes", "Kale and tofu curry with rice", "Methi dal with whole wheat roti"],
                            ["Nut butter toast", "Vegan vegetable pulao", "Okra and lentil stir-fry"],
                            ["Banana smoothie", "Spinach and chickpea curry", "Pumpkin and lentil soup"]
                        ]
                    elif prediction_result == "Diabetic":
                        weekly_meals = [
                            ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                            ["Chia pudding with nuts", "Lauki dal with chapati", "Palak tofu with quinoa"],
                            ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                            ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                            ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                            ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                            ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                        ]
                
    
    if bmi_category == "Normal":
        if gender == "Female":
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Green smoothie with chia", "Quinoa and black bean salad", "Grilled tofu with vegetables"],
                    ["Overnight oats with almond milk", "Lentil soup with whole wheat bread", "Stuffed bell peppers"],
                    ["Fruit and nut parfait", "Steamed broccoli with chickpeas", "Vegetable stir-fry with tofu"],
                    ["Chia pudding with flaxseeds", "Brown rice with mixed vegetables", "Baked sweet potato with tahini"],
                    ["Vegan pancakes with berries", "Tofu stir-fry with quinoa", "Dal with millet roti"],
                    ["Soy yogurt with nuts", "Vegetable biryani with raita", "Lentil-stuffed zucchini"],
                    ["Banana smoothie with seeds", "Chickpea and kale stew", "Grilled eggplant with tahini"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almond milk", "Vegetable soup with lentils", "Grilled tempeh with sautéed greens"],
                    ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                    ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                    ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                    ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                    ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                    ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                    ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                    ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                    ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                    ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                    ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                    ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                ]
            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Vegetable upma", "Palak paneer with chapati", "Dal with brown rice"],
                    ["Besan cheela with mint chutney", "Vegetable pulao with curd", "Stuffed capsicum with roti"],
                    ["Banana smoothie with seeds", "Chickpea salad with lemon dressing", "Methi thepla with yogurt"],
                    ["Oats porridge with flaxseeds", "Dal khichdi with ghee", "Rajma with whole wheat roti"],
                    ["Idli with sambhar", "Vegetable stir-fry with tofu", "Lauki sabzi with chapati"],
                    ["Moong dal dosa with chutney", "Bhindi masala with brown rice", "Baingan bharta with roti"],
                    ["Sprouts chaat", "Curd rice with pickle", "Pumpkin curry with chapati"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                    ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                    ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                    ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                    ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                    ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                    ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                    ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                    ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                    ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                    ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                    ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                    ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                ]
            if diet == "Non-Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Boiled eggs with toast", "Chicken curry with brown rice", "Grilled fish with vegetables"],
                    ["Oats with honey", "Grilled chicken salad", "Mutton curry with chapati"],
                    ["Scrambled eggs with avocado", "Fish curry with rice", "Chicken stir-fry with quinoa"],
                    ["Greek yogurt with nuts", "Prawn biryani", "Tandoori chicken with sautéed greens"],
                    ["Banana smoothie with whey", "Egg curry with whole wheat roti", "Grilled salmon with broccoli"],
                    ["Paneer bhurji with toast", "Chicken stew with multigrain bread", "Fish tikka with vegetable stir-fry"],
                    ["Sprouts chaat with eggs", "Mutton soup with brown rice", "Grilled prawns with quinoa"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Grilled chicken with steamed vegetables", "Fish curry with barley"],
                    ["Boiled eggs with nuts", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Almond smoothie", "Mutton soup with multigrain toast", "Grilled salmon with sautéed greens"],
                    ["Chia pudding", "Egg curry with chapati", "Chicken tikka with salad"],
                    ["Peanut butter toast", "Fish stew with whole wheat bread", "Grilled chicken with lentils"],
                    ["Protein shake", "Vegetable omelet with toast", "Tandoori fish with stir-fried vegetables"],
                    ["Fruit bowl with nuts", "Chicken biryani with raita", "Prawn curry with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled chicken with greens", "Fish curry with quinoa"],
                    ["Scrambled eggs with flaxseeds", "Grilled salmon with stir-fried vegetables", "Mutton stew with brown rice"],
                    ["Almond smoothie", "Egg curry with sautéed spinach", "Chicken stew with barley"],
                    ["Greek yogurt with nuts", "Prawn curry with whole wheat roti", "Grilled chicken with salad"],
                    ["Boiled eggs with peanut butter toast", "Tandoori fish with vegetables", "Mutton soup with chapati"],
                    ["Oats with chia seeds", "Chicken biryani with cucumber raita", "Fish tikka with quinoa"],
                    ["Flaxseed smoothie", "Vegetable omelet with toast", "Grilled prawns with stir-fried greens"]
                ]
                    
        if gender == "Male":
            if diet == "Non-Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Boiled eggs with toast", "Chicken curry with brown rice", "Grilled fish with vegetables"],
                    ["Oats with honey", "Grilled chicken salad", "Mutton curry with chapati"],
                    ["Scrambled eggs with avocado", "Fish curry with rice", "Chicken stir-fry with quinoa"],
                    ["Greek yogurt with nuts", "Prawn biryani", "Tandoori chicken with sautéed greens"],
                    ["Banana smoothie with whey", "Egg curry with whole wheat roti", "Grilled salmon with broccoli"],
                    ["Paneer bhurji with toast", "Chicken stew with multigrain bread", "Fish tikka with vegetable stir-fry"],
                    ["Sprouts chaat with eggs", "Mutton soup with brown rice", "Grilled prawns with quinoa"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Grilled chicken with steamed vegetables", "Fish curry with barley"],
                    ["Boiled eggs with nuts", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Almond smoothie", "Mutton soup with multigrain toast", "Grilled salmon with sautéed greens"],
                    ["Chia pudding", "Egg curry with chapati", "Chicken tikka with salad"],
                    ["Peanut butter toast", "Fish stew with whole wheat bread", "Grilled chicken with lentils"],
                    ["Protein shake", "Vegetable omelet with toast", "Tandoori fish with stir-fried vegetables"],
                    ["Fruit bowl with nuts", "Chicken biryani with raita", "Prawn curry with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled chicken with greens", "Fish curry with quinoa"],
                    ["Scrambled eggs with flaxseeds", "Grilled salmon with stir-fried vegetables", "Mutton stew with brown rice"],
                    ["Almond smoothie", "Egg curry with sautéed spinach", "Chicken stew with barley"],
                    ["Greek yogurt with nuts", "Prawn curry with whole wheat roti", "Grilled chicken with salad"],
                    ["Boiled eggs with peanut butter toast", "Tandoori fish with vegetables", "Mutton soup with chapati"],
                    ["Oats with chia seeds", "Chicken biryani with cucumber raita", "Fish tikka with quinoa"],
                    ["Flaxseed smoothie", "Vegetable omelet with toast", "Grilled prawns with stir-fried greens"]
                ]
                    
            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Oats with almonds", "Dal with brown rice", "Vegetable stir-fry with chapati"],
                    ["Banana smoothie with chia seeds", "Paneer curry with quinoa", "Stuffed bell peppers"],
                    ["Sprouts chaat", "Rajma with multigrain roti", "Grilled tofu with sautéed greens"],
                    ["Greek yogurt with nuts", "Mixed vegetable biryani", "Lentil soup with whole wheat bread"],
                    ["Poha with peanuts", "Chickpea curry with brown rice", "Palak paneer with chapati"],
                    ["Moong dal cheela", "Vegetable khichdi", "Stuffed paratha with curd"],
                    ["Ragi porridge", "Sambar with rice", "Baingan bharta with roti"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Dal with brown rice", "Stir-fried vegetables with chapati"],
                    ["Sprouts salad", "Chole with multigrain roti", "Grilled paneer with sautéed spinach"],
                    ["Almond smoothie", "Mixed vegetable curry with quinoa", "Lauki sabzi with roti"],
                    ["Chia pudding", "Vegetable pulao with raita", "Dal tadka with brown rice"],
                    ["Ragi dosa with chutney", "Moong dal with chapati", "Grilled tofu with vegetables"],
                    ["Peanut butter toast", "Sambar with rice", "Besan cheela with curd"],
                    ["Flaxseed smoothie", "Rajma with quinoa", "Stuffed capsicum with multigrain roti"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled dal with greens", "Vegetable curry with quinoa"],
                    ["Scrambled paneer with flaxseeds", "Grilled tofu with stir-fried vegetables", "Lentil soup with brown rice"],
                    ["Almond smoothie", "Bhindi masala with chapati", "Sprouts curry with quinoa"],
                    ["Greek yogurt with nuts", "Lauki sabzi with roti", "Moong dal cheela with curd"],
                    ["Ragi dosa with peanut chutney", "Tinda curry with chapati", "Soya bhurji with multigrain toast"],
                    ["Oats with chia seeds", "Sambar with brown rice", "Grilled vegetables with hummus"],
                    ["Flaxseed smoothie", "Stuffed paratha with low-fat curd", "Besan cheela with sautéed greens"]
                ]
                    
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                ["Oats with almond milk and flaxseeds", "Dal with brown rice", "Grilled vegetables with hummus"],
                ["Banana smoothie with chia seeds", "Chickpea curry with quinoa", "Stuffed bell peppers with tofu"],
                ["Sprouts chaat", "Rajma with multigrain roti", "Sautéed mushrooms with garlic and herbs"],
                ["Chia pudding with walnuts", "Vegetable pulao with raita", "Lentil soup with whole wheat bread"],
                ["Poha with peanuts", "Chole with brown rice", "Baingan bharta with chapati"],
                ["Ragi dosa with coconut chutney", "Vegetable khichdi", "Stuffed paratha with almond curd"],
                ["Quinoa porridge", "Sambar with millet rice", "Stir-fried greens with tofu and chapati"]
            ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                ["Oats with chia seeds", "Dal with red rice", "Mixed vegetable stir-fry with quinoa"],
                ["Sprouts salad", "Chickpea curry with multigrain roti", "Tofu scramble with spinach"],
                ["Flaxseed smoothie", "Bottle gourd sabzi with roti", "Lauki chana dal with rice"],
                ["Greek-style almond yogurt with walnuts", "Vegetable pulao with raita", "Dal tadka with brown rice"],
                ["Ragi dosa with mint chutney", "Mixed lentil soup with chapati", "Stuffed tomatoes with tofu"],
                ["Whole wheat toast with peanut butter", "Sambar with millet rice", "Besan cheela with sautéed greens"],
                ["Green smoothie (spinach, chia, banana)", "Rajma with quinoa", "Stuffed capsicum with lentils"]
            ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                ["Chia pudding with walnuts", "Boiled moong dal with greens", "Vegetable curry with quinoa"],
                ["Tofu scramble with flaxseeds", "Grilled mushrooms with stir-fried vegetables", "Lentil soup with brown rice"],
                ["Almond milk smoothie", "Bhindi masala with chapati", "Sprouts curry with quinoa"],
                ["Chia yogurt with nuts", "Lauki sabzi with millet roti", "Moong dal cheela with almond curd"],
                ["Ragi dosa with peanut chutney", "Tinda curry with chapati", "Soya bhurji with multigrain toast"],
                ["Oats with chia seeds", "Sambar with foxtail millet rice", "Grilled vegetables with hummus"],
                ["Flaxseed smoothie", "Stuffed paratha with low-fat soy curd", "Besan cheela with sautéed greens"]
            ]

    if bmi_category == "Overweight":
        if gender == "Male":
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Oats with almond milk and chia", "Dal with brown rice", "Grilled vegetables with hummus"],
                    ["Green smoothie with flaxseeds", "Chickpea curry with quinoa", "Stuffed bell peppers with tofu"],
                    ["Sprouts chaat", "Rajma with multigrain roti", "Sautéed mushrooms with garlic and herbs"],
                    ["Chia pudding with walnuts", "Vegetable pulao with raita", "Lentil soup with whole wheat bread"],
                    ["Poha with peanuts", "Chole with brown rice", "Baingan bharta with chapati"],
                    ["Ragi dosa with coconut chutney", "Vegetable khichdi", "Stuffed paratha with almond curd"],
                    ["Quinoa porridge", "Sambar with millet rice", "Stir-fried greens with tofu and chapati"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with chia seeds", "Dal with red rice", "Mixed vegetable stir-fry with quinoa"],
                    ["Sprouts salad", "Chickpea curry with multigrain roti", "Tofu scramble with spinach"],
                    ["Flaxseed smoothie", "Bottle gourd sabzi with roti", "Lauki chana dal with rice"],
                    ["Greek-style almond yogurt with walnuts", "Vegetable pulao with raita", "Dal tadka with brown rice"],
                    ["Ragi dosa with mint chutney", "Mixed lentil soup with chapati", "Stuffed tomatoes with tofu"],
                    ["Whole wheat toast with peanut butter", "Sambar with millet rice", "Besan cheela with sautéed greens"],
                    ["Green smoothie (spinach, chia, banana)", "Rajma with quinoa", "Stuffed capsicum with lentils"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with walnuts", "Boiled moong dal with greens", "Vegetable curry with quinoa"],
                    ["Tofu scramble with flaxseeds", "Grilled mushrooms with stir-fried vegetables", "Lentil soup with brown rice"],
                    ["Almond milk smoothie", "Bhindi masala with chapati", "Sprouts curry with quinoa"],
                    ["Chia yogurt with nuts", "Lauki sabzi with millet roti", "Moong dal cheela with almond curd"],
                    ["Ragi dosa with peanut chutney", "Tinda curry with chapati", "Soya bhurji with multigrain toast"],
                    ["Oats with chia seeds", "Sambar with foxtail millet rice", "Grilled vegetables with hummus"],
                    ["Flaxseed smoothie", "Stuffed paratha with low-fat soy curd", "Besan cheela with sautéed greens"]
                ]

            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Vegetable upma", "Palak paneer with chapati", "Dal with brown rice"],
                    ["Besan cheela with mint chutney", "Vegetable pulao with curd", "Stuffed capsicum with roti"],
                    ["Banana smoothie with seeds", "Chickpea salad with lemon dressing", "Methi thepla with yogurt"],
                    ["Oats porridge with flaxseeds", "Dal khichdi with ghee", "Rajma with whole wheat roti"],
                    ["Idli with sambhar", "Vegetable stir-fry with paneer", "Lauki sabzi with chapati"],
                    ["Moong dal dosa with chutney", "Bhindi masala with brown rice", "Baingan bharta with roti"],
                    ["Sprouts chaat", "Curd rice with pickle", "Pumpkin curry with chapati"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                    ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                    ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                    ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                    ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                    ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                    ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                    ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                    ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                    ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                    ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                    ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                    ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                ]
            if diet == "Non-Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Boiled eggs with whole wheat toast", "Grilled chicken with brown rice", "Fish curry with chapati"],
                    ["Oats with nuts", "Egg curry with multigrain roti", "Grilled salmon with sautéed vegetables"],
                    ["Scrambled eggs with avocado", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Greek yogurt with nuts", "Mutton stew with multigrain bread", "Grilled fish with salad"],
                    ["Banana smoothie with whey protein", "Chicken curry with chapati", "Grilled prawns with quinoa"],
                    ["Paneer omelet with toast", "Tandoori chicken with vegetables", "Fish tikka with stir-fried greens"],
                    ["Sprouts chaat with eggs", "Mutton soup with brown rice", "Grilled chicken with spinach"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Grilled chicken with steamed vegetables", "Fish curry with barley"],
                    ["Boiled eggs with nuts", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Almond smoothie", "Mutton soup with multigrain toast", "Grilled salmon with sautéed greens"],
                    ["Chia pudding", "Egg curry with chapati", "Chicken tikka with salad"],
                    ["Peanut butter toast", "Fish stew with whole wheat bread", "Grilled chicken with lentils"],
                    ["Protein shake", "Vegetable omelet with toast", "Tandoori fish with stir-fried vegetables"],
                    ["Fruit bowl with nuts", "Chicken biryani with raita", "Prawn curry with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled chicken with greens", "Fish curry with quinoa"],
                    ["Scrambled eggs with flaxseeds", "Grilled salmon with stir-fried vegetables", "Mutton stew with brown rice"],
                    ["Almond smoothie", "Egg curry with sautéed spinach", "Chicken stew with barley"],
                    ["Greek yogurt with nuts", "Prawn curry with whole wheat roti", "Grilled chicken with salad"],
                    ["Boiled eggs with peanut butter toast", "Tandoori fish with vegetables", "Mutton soup with chapati"],
                    ["Oats with chia seeds", "Chicken biryani with cucumber raita", "Fish tikka with quinoa"],
                    ["Flaxseed smoothie", "Vegetable omelet with toast", "Grilled prawns with stir-fried greens"]
                ]
                    
                    
        if gender == "Female":
            if diet == "Non-Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Boiled eggs with multigrain toast", "Grilled fish with brown rice", "Chicken curry with chapati"],
                    ["Oats with almonds", "Egg salad with whole wheat bread", "Grilled salmon with sautéed vegetables"],
                    ["Scrambled eggs with avocado", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Greek yogurt with nuts", "Mutton stew with mixed vegetables", "Grilled fish with salad"],
                    ["Banana smoothie with flaxseeds", "Chicken curry with chapati", "Grilled prawns with sautéed spinach"],
                    ["Paneer omelet with toast", "Tandoori chicken with steamed vegetables", "Fish tikka with stir-fried greens"],
                    ["Sprouts chaat with eggs", "Mutton soup with brown rice", "Grilled chicken with quinoa"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Grilled chicken with steamed vegetables", "Fish curry with barley"],
                    ["Boiled eggs with nuts", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Almond smoothie", "Mutton soup with multigrain toast", "Grilled salmon with sautéed greens"],
                    ["Chia pudding", "Egg curry with chapati", "Chicken tikka with salad"],
                    ["Peanut butter toast", "Fish stew with whole wheat bread", "Grilled chicken with lentils"],
                    ["Protein shake", "Vegetable omelet with toast", "Tandoori fish with stir-fried vegetables"],
                    ["Fruit bowl with nuts", "Chicken biryani with raita", "Prawn curry with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled chicken with greens", "Fish curry with quinoa"],
                    ["Scrambled eggs with flaxseeds", "Grilled salmon with stir-fried vegetables", "Mutton stew with brown rice"],
                    ["Almond smoothie", "Egg curry with sautéed spinach", "Chicken stew with barley"],
                    ["Greek yogurt with nuts", "Prawn curry with whole wheat roti", "Grilled chicken with salad"],
                    ["Boiled eggs with peanut butter toast", "Tandoori fish with vegetables", "Mutton soup with chapati"],
                    ["Oats with chia seeds", "Chicken biryani with cucumber raita", "Fish tikka with quinoa"],
                    ["Flaxseed smoothie", "Vegetable omelet with toast", "Grilled prawns with stir-fried greens"]
                ]
                    
            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Oats with chia seeds", "Vegetable curry with quinoa", "Palak paneer with chapati"],
                    ["Greek yogurt with almonds", "Lentil soup with whole wheat bread", "Vegetable stir-fry with tofu"],
                    ["Chia pudding with flaxseeds", "Mixed vegetable biryani", "Moong dal with brown rice"],
                    ["Banana smoothie with flaxseeds", "Rajma with chapati", "Vegetable upma"],
                    ["Overnight oats with berries", "Chickpea curry with quinoa", "Grilled tofu with sautéed spinach"],
                    ["Peanut butter toast", "Chickpea and vegetable stew", "Paneer tikka with salad"],
                    ["Mango smoothie with seeds", "Vegetable khichdi", "Grilled vegetables with hummus"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Vegetable soup with lentils", "Grilled tempeh with sautéed greens"],
                    ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                    ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                    ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                    ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                    ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                    ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                    ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                    ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                    ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                    ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                    ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                    ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                ]
                    
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Green smoothie with chia", "Quinoa and black bean salad", "Grilled tofu with vegetables"],
                    ["Overnight oats with almond milk", "Lentil soup with whole wheat bread", "Stuffed bell peppers"],
                    ["Fruit and nut parfait", "Steamed broccoli with chickpeas", "Vegetable stir-fry with tofu"],
                    ["Chia pudding with flaxseeds", "Brown rice with mixed vegetables", "Baked sweet potato with tahini"],
                    ["Vegan pancakes with berries", "Tofu stir-fry with quinoa", "Dal with millet roti"],
                    ["Soy yogurt with nuts", "Vegetable biryani with raita", "Lentil-stuffed zucchini"],
                    ["Banana smoothie with seeds", "Chickpea and kale stew", "Grilled eggplant with tahini"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almond milk", "Vegetable soup with lentils", "Grilled tempeh with sautéed greens"],
                    ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                    ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                    ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                    ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                    ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                    ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                    ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                    ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                    ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                    ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                    ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                    ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                ]
                    
    if bmi_category == "Underweight":
        if gender == "Female":
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Green smoothie with chia", "Quinoa and chickpea salad", "Grilled tofu with vegetables"],
                    ["Overnight oats with almond milk", "Lentil soup with whole wheat bread", "Stuffed bell peppers"],
                    ["Fruit and nut parfait", "Steamed broccoli with chickpeas", "Vegetable stir-fry with tofu"],
                    ["Chia pudding with flaxseeds", "Brown rice with mixed vegetables", "Baked sweet potato with tahini"],
                    ["Vegan pancakes with berries", "Tofu stir-fry with quinoa", "Dal with millet roti"],
                    ["Soy yogurt with nuts", "Vegetable biryani with raita", "Lentil-stuffed zucchini"],
                    ["Banana smoothie with seeds", "Chickpea and kale stew", "Grilled eggplant with tahini"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almond milk", "Vegetable soup with lentils", "Grilled tempeh with sautéed greens"],
                    ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                    ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                    ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                    ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                    ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                    ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                    ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                    ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                    ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                    ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                    ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                    ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                ]
                    
            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Vegetable upma", "Palak paneer with chapati", "Dal with brown rice"],
                    ["Besan cheela with mint chutney", "Vegetable pulao with curd", "Stuffed capsicum with roti"],
                    ["Banana smoothie with seeds", "Chickpea salad with lemon dressing", "Methi thepla with yogurt"],
                    ["Oats porridge with flaxseeds", "Dal khichdi with ghee", "Rajma with whole wheat roti"],
                    ["Idli with sambhar", "Vegetable stir-fry with tofu", "Lauki sabzi with chapati"],
                    ["Moong dal dosa with chutney", "Bhindi masala with brown rice", "Baingan bharta with roti"],
                    ["Sprouts chaat", "Curd rice with pickle", "Pumpkin curry with chapati"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                    ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                    ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                    ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                    ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                    ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                    ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                    ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                    ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                    ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                    ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                    ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                    ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                ]
                    
            if diet == "Non-Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Boiled eggs with toast", "Chicken curry with brown rice", "Grilled fish with vegetables"],
                    ["Oats with honey", "Grilled chicken salad", "Mutton curry with chapati"],
                    ["Scrambled eggs with avocado", "Fish curry with rice", "Chicken stir-fry with quinoa"],
                    ["Greek yogurt with nuts", "Prawn biryani", "Tandoori chicken with sautéed greens"],
                    ["Banana smoothie with whey", "Egg curry with whole wheat roti", "Grilled salmon with broccoli"],
                    ["Paneer bhurji with toast", "Chicken stew with multigrain bread", "Fish tikka with vegetable stir-fry"],
                    ["Sprouts chaat with eggs", "Mutton soup with brown rice", "Grilled prawns with quinoa"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Grilled chicken with steamed vegetables", "Fish curry with barley"],
                    ["Boiled eggs with nuts", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Almond smoothie", "Mutton soup with multigrain toast", "Grilled salmon with sautéed greens"],
                    ["Chia pudding", "Egg curry with chapati", "Chicken tikka with salad"],
                    ["Peanut butter toast", "Fish stew with whole wheat bread", "Grilled chicken with lentils"],
                    ["Protein shake", "Vegetable omelet with toast", "Tandoori fish with stir-fried vegetables"],
                    ["Fruit bowl with nuts", "Chicken biryani with raita", "Prawn curry with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled chicken with greens", "Fish curry with quinoa"],
                    ["Scrambled eggs with flaxseeds", "Grilled salmon with stir-fried vegetables", "Mutton stew with brown rice"],
                    ["Almond smoothie", "Egg curry with sautéed spinach", "Chicken stew with barley"],
                    ["Greek yogurt with nuts", "Prawn curry with whole wheat roti", "Grilled chicken with salad"],
                    ["Boiled eggs with peanut butter toast", "Tandoori fish with vegetables", "Mutton soup with chapati"],
                    ["Oats with chia seeds", "Chicken biryani with cucumber raita", "Fish tikka with quinoa"],
                    ["Flaxseed smoothie", "Vegetable omelet with toast", "Grilled prawns with stir-fried greens"]
                ]
                    
        if gender == "Male":
            if diet == "Non-Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Boiled eggs with toast", "Chicken curry with brown rice", "Grilled fish with vegetables"],
                    ["Oats with honey", "Grilled chicken salad", "Mutton curry with chapati"],
                    ["Scrambled eggs with avocado", "Fish curry with rice", "Chicken stir-fry with quinoa"],
                    ["Greek yogurt with nuts", "Prawn biryani", "Tandoori chicken with sautéed greens"],
                    ["Banana smoothie with whey", "Egg curry with whole wheat roti", "Grilled salmon with broccoli"],
                    ["Paneer bhurji with toast", "Chicken stew with multigrain bread", "Fish tikka with vegetable stir-fry"],
                    ["Sprouts chaat with eggs", "Mutton soup with brown rice", "Grilled prawns with quinoa"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with flaxseeds", "Grilled chicken with steamed vegetables", "Fish curry with barley"],
                    ["Boiled eggs with nuts", "Chicken stir-fry with quinoa", "Prawn curry with brown rice"],
                    ["Almond smoothie", "Mutton soup with multigrain toast", "Grilled salmon with sautéed greens"],
                    ["Chia pudding", "Egg curry with chapati", "Chicken tikka with salad"],
                    ["Peanut butter toast", "Fish stew with whole wheat bread", "Grilled chicken with lentils"],
                    ["Protein shake", "Vegetable omelet with toast", "Tandoori fish with stir-fried vegetables"],
                    ["Fruit bowl with nuts", "Chicken biryani with raita", "Prawn curry with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia seed pudding", "Boiled chicken with greens", "Fish curry with quinoa"],
                    ["Scrambled eggs with flaxseeds", "Grilled salmon with stir-fried vegetables", "Mutton stew with brown rice"],
                    ["Almond smoothie", "Egg curry with sautéed spinach", "Chicken stew with barley"],
                    ["Greek yogurt with nuts", "Prawn curry with whole wheat roti", "Grilled chicken with salad"],
                    ["Boiled eggs with peanut butter toast", "Tandoori fish with vegetables", "Mutton soup with chapati"],
                    ["Oats with chia seeds", "Chicken biryani with cucumber raita", "Fish tikka with quinoa"],
                    ["Flaxseed smoothie", "Vegetable omelet with toast", "Grilled prawns with stir-fried greens"]
                ]
                    
            if diet == "Veg":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Vegetable upma", "Palak paneer with chapati", "Dal with brown rice"],
                    ["Besan cheela with mint chutney", "Vegetable pulao with curd", "Stuffed capsicum with roti"],
                    ["Banana smoothie with seeds", "Chickpea salad with lemon dressing", "Methi thepla with yogurt"],
                    ["Oats porridge with flaxseeds", "Dal khichdi with ghee", "Rajma with whole wheat roti"],
                    ["Idli with sambhar", "Vegetable stir-fry with tofu", "Lauki sabzi with chapati"],
                    ["Moong dal dosa with chutney", "Bhindi masala with brown rice", "Baingan bharta with roti"],
                    ["Sprouts chaat", "Curd rice with pickle", "Pumpkin curry with chapati"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almonds", "Dal khichdi with ghee", "Lauki sabzi with roti"],
                    ["Sprouts chaat", "Bhindi masala with rice", "Palak paneer with chapati"],
                    ["Methi paratha", "Rajma with brown rice", "Vegetable stew with roti"],
                    ["Vegetable poha", "Lentil soup with bread", "Karela sabzi with dal"],
                    ["Banana shake with nuts", "Tofu stir-fry with quinoa", "Ghiya curry with roti"],
                    ["Besan cheela", "Vegetable pulao with curd", "Dal fry with tandoori roti"],
                    ["Cucumber sandwich", "Chana masala with roti", "Kadhi with brown rice"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Ragi dosa with chutney", "Moong dal khichdi", "Bitter gourd stir-fry with roti"],
                    ["Chia pudding with nuts", "Lauki dal with chapati", "Palak paneer with quinoa"],
                    ["Besan cheela with mint chutney", "Brown rice with dal", "Stuffed brinjal with roti"],
                    ["Sprouts salad", "Vegetable daliya", "Methi thepla with curd"],
                    ["Oats upma", "Karela dal with rice", "Ghiya curry with chapati"],
                    ["Flaxseed smoothie", "Rajma with brown rice", "Sautéed okra with dal"],
                    ["Peanut butter toast", "Bajra roti with mixed vegetable curry", "Masoor dal with tandoori roti"]
                ]
                    
                    
            if diet == "Vegan":
                if prediction_result == "Non-Diabetic":
                    weekly_meals = [
                    ["Green smoothie with chia", "Quinoa and black bean salad", "Grilled tofu with vegetables"],
                    ["Overnight oats with almond milk", "Lentil soup with whole wheat bread", "Stuffed bell peppers"],
                    ["Fruit and nut parfait", "Steamed broccoli with chickpeas", "Vegetable stir-fry with tofu"],
                    ["Chia pudding with flaxseeds", "Brown rice with mixed vegetables", "Baked sweet potato with tahini"],
                    ["Vegan pancakes with berries", "Tofu stir-fry with quinoa", "Dal with millet roti"],
                    ["Soy yogurt with nuts", "Vegetable biryani with raita", "Lentil-stuffed zucchini"],
                    ["Banana smoothie with seeds", "Chickpea and kale stew", "Grilled eggplant with tahini"]
                ]
                elif prediction_result == "Pre-Diabetic":
                    weekly_meals = [
                    ["Oats with almond milk", "Vegetable soup with lentils", "Grilled tempeh with sautéed greens"],
                    ["Smoothie with spinach and banana", "Brown rice with stir-fried vegetables", "Stuffed mushrooms"],
                    ["Multigrain toast with avocado", "Grilled tempeh with salad", "Dal with whole wheat chapati"],
                    ["Almond butter with oats", "Steamed vegetables with quinoa", "Vegan curry with chickpeas"],
                    ["Soy yogurt with nuts", "Mushroom and lentil stew", "Tofu and vegetable stir-fry"],
                    ["Vegan protein shake", "Whole wheat pasta with tomato sauce", "Sweet potato and black bean tacos"],
                    ["Fruit salad with flaxseeds", "Lentil and spinach soup", "Quinoa with roasted vegetables"]
                ]
                elif prediction_result == "Diabetic":
                    weekly_meals = [
                    ["Chia pudding with flaxseeds", "Steamed broccoli and lentils", "Tofu and bell pepper stir-fry"],
                    ["Green smoothie with nuts", "Grilled zucchini with hummus", "Lentil soup with barley"],
                    ["Oats with cinnamon", "Quinoa and vegetable pilaf", "Sautéed kale with garlic"],
                    ["Vegan protein shake", "Mixed greens with chickpeas", "Stuffed bell peppers with lentils"],
                    ["Tofu scramble with spinach", "Brown rice with vegetable curry", "Grilled eggplant with tahini"],
                    ["Almond yogurt with nuts", "Whole grain wrap with hummus", "Steamed vegetables with quinoa"],
                    ["Berry smoothie with chia", "Mushroom soup with lentils", "Vegetable stir-fry with tempeh"]
                ]
          
    
      
                    
    if not weekly_meals:
        print("❌ No suitable meal plan found!")
        return None
    
    return weekly_meals

@app.route('/account')
def account():
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in first.', 'danger')
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()

    conn = sqlite3.connect('predict.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE user_id = ?', (user_id,))
    predictions = cursor.fetchall()
    conn.close()

    return render_template('account.html', user=user, predictions=predictions)


@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')


# Function to fetch user health details
def get_user_health_report(user_id):
    
    conn_users = sqlite3.connect("users.db")
    cursor_users = conn_users.cursor()
    cursor_users.execute("SELECT username, email FROM users WHERE id = ?", (user_id,))
    user = cursor_users.fetchone()
    conn_users.close()
    
    conn_predict = sqlite3.connect("predict.db")
    cursor_predict = conn_predict.cursor()
    cursor_predict.execute("SELECT age, bmi, prediction FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
    prediction = cursor_predict.fetchone()
    conn_predict.close()
    
    conn_meal = sqlite3.connect("meal.db")
    cursor_meal = conn_meal.cursor()
    cursor_meal.execute("""
            SELECT day, breakfast, lunch, dinner 
            FROM meal 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 7
        """, (user_id,))
    meals = cursor_meal.fetchall()
    
    meal_plan = [] if not meals else [
        {
            "Day": meal[0],
            "Breakfast": meal[1],
            "Lunch": meal[2],
            "Dinner": meal[3],
        }
        for meal in meals
    ]
    
    meal_status = "Meal plan available" if meals else "Meal plan not available"
    conn_meal.close()
    
    conn_exercise = sqlite3.connect("exercise.db")
    cursor_exercise = conn_exercise.cursor()
    cursor_exercise.execute("""
        SELECT workout_type, frequency, duration, exercises 
        FROM exercise 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (user_id,))
    exercise = cursor_exercise.fetchone()
    conn_exercise.close()
    
    # Debugging Print Statements
    print(f"User Data: {user}")
    print(f"Prediction Data: {prediction}")
    #print(f"Meal Plan Data: {meal}")
    print(f"Exercise Data: {exercise}")
    
    return {
        "Name": user[0] if user else "N/A",
        "Email": user[1] if user else "N/A",
        "Age": prediction[0] if prediction else "N/A",
        "BMI": prediction[1] if prediction else "N/A",
        "Prediction": prediction[2] if prediction else "N/A",
        "Meal Plan Status":meal_status,
        #"Meal Plan": meal_plan if meal_plan else "No meal plan available",
        "Workout Type": exercise[0] if exercise else "N/A",
        "Frequency": exercise[1] if exercise else "N/A",
        "Duration": exercise[2] if exercise else "N/A",
        "Exercises": exercise[3] if exercise else "N/A"
    }

# Route to generate and download PDF
@app.route('/download_report')
def download_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    report_data = get_user_health_report(user_id)
    
    from reportlab.platypus import SimpleDocTemplate
    
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    
    from reportlab.lib.styles import getSampleStyleSheet
    
    styles = getSampleStyleSheet()
    elements = []
    
    from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image
    import os
    
    # Add Logo (Optional)
    logo_path = "static/logo2.png"
    if os.path.exists(logo_path):
        elements.append(Image(logo_path, width=80, height=50))
        elements.append(Spacer(1, 10))

    # Personalized Title
    user_name = report_data.get("Name", "User")  # Default fallback if name not found
    report_title = f"{user_name}'s Health Report"
    elements.append(Paragraph(report_title, styles['Title']))
    elements.append(Spacer(1, 10))

    # Convert Data to Table Format
    table_data = [["Category", "Details"]]  # Table header
    table_data.extend([[key, str(value)] for key, value in report_data.items()])  # Add user data
    
    # Table Styling
    table = Table(table_data, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))
    
    
    # Format and Display Meal Plan Properly
    #meal_plan = report_data.get("Meal Plan", [])
    
    # Fetch meal plan separately from the database
    conn_meal = sqlite3.connect("meal.db")
    cursor_meal = conn_meal.cursor()
    cursor_meal.execute("""
    SELECT day, breakfast, lunch, dinner 
    FROM meal 
    WHERE user_id = ? 
    ORDER BY timestamp DESC 
    LIMIT 7
    """, (user_id,))
    meals = cursor_meal.fetchall()
    conn_meal.close()

    meal_plan = [] if not meals else [
    {
        "Day": meal[0],
        "Breakfast": meal[1],
        "Lunch": meal[2],
        "Dinner": meal[3],
    }
    for meal in meals
    ]

    if meal_plan:
    #if meal_plan and meal_plan != "No meal plan available":
        elements.append(Paragraph("Weekly Meal Plan", styles['Heading2']))
        elements.append(Spacer(1, 10))

        # Meal Table Headers
        meal_table_data = [["Day", "Meal Type", "Meal"]]
        
        # Insert Meals row-wise
        for meal in meal_plan:
            meal_table_data.append([meal["Day"], "Breakfast", meal["Breakfast"]])
            meal_table_data.append(["", "Lunch", meal["Lunch"]])
            meal_table_data.append(["", "Dinner", meal["Dinner"]])
            meal_table_data.append(["", "", ""])  # Empty row for spacing
        
        # Meal Plan Table Styling
        meal_table = Table(meal_table_data, colWidths=[100, 100, 250])
        meal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(meal_table)
        elements.append(Spacer(1, 20))
    
    from datetime import datetime
    # Footer with Date
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    
    #from reportlab.pdfgen import canvas
    # Build PDF properly
    pdf.build(elements)
    #pdf.save()
    buffer.seek(0)
    
    user_name = report_data.get("Name", "User").replace(" ", "_")

    download_name = f"{user_name}_Health_Report.pdf"
    return send_file(buffer, as_attachment=True, download_name=download_name, mimetype='application/pdf')


@app.route('/faq')
def faq():
    return render_template('faq.html')
# Feedback route
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        user_id = session.get('user_id')  # Get the logged-in user's ID
        print("User ID:", user_id) 
        if not user_id:
            flash('You must be logged in to send feedback!', 'danger')
            return redirect(url_for('login'))

        if feedback_text:
            conn = sqlite3.connect('feedback.db')
            c = conn.cursor()

            # Insert feedback, ensuring the same user_id for multiple feedbacks
            c.execute('INSERT INTO feedback (user_id, feedback) VALUES (?, ?)', (user_id, feedback_text))
            conn.commit()
            conn.close()

            flash("Thank you for your feedback!", "success")
            return redirect(url_for('mainpage2'))  # Redirect to mainpage2 after sending feedback

    return render_template('feedback.html')

from werkzeug.security import generate_password_hash, check_password_hash

# Function to get user details
def get_user_details(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT username, email FROM users WHERE id=?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

# Route for settings page
@app.route('/setting', methods=['GET', 'POST'])
def setting():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    user_id = session['user_id']
    user = get_user_details(user_id)

    if request.method == 'POST':
        # Handling username and email update
        if 'username' in request.form and 'email' in request.form:
            new_username = request.form['username']
            new_email = request.form['email']

            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET username=?, email=? WHERE id=?", (new_username, new_email, user_id))
            conn.commit()
            conn.close()

            flash("Account details updated successfully!", "success")
            return redirect(url_for('setting'))

        


    return render_template('setting.html', user=user)

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return jsonify({"success": False, "error": "User not logged in"}), 401

    user_id = session.get('user_id')
    print("Session User ID:", session.get('user_id'))  # Debugging

    print("Form Data:", request.form)  # Debugging

    current_password = request.form.get('current_password') 
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    

    if not all([current_password, new_password, confirm_password]):
        return jsonify({"success": False, "error": "Missing fields"}), 400
    
    # Check if new passwords match
    if new_password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match"}), 400


    # Enforce password security (example: at least 8 characters)
    if len(new_password) < 8:
        return jsonify({"success": False, "error": "New password must be at least 8 characters long"}), 400
    
    # Retrieve stored password hash securely
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()

        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404

        stored_password = user[0]

    
    from werkzeug.security import generate_password_hash, check_password_hash

    
    # Verify current password
    if not check_password_hash(stored_password, current_password):
        conn.close()
        return jsonify({"success": False, "error": "Incorrect current password"}), 400

    # Prevent users from reusing the same password
    if check_password_hash(stored_password, new_password):
        return jsonify({"success": False, "error": "New password cannot be the same as the old password"}), 400
        
    # Hash the new password
    hashed_new_password = generate_password_hash(new_password)

    # Update password in database
    cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_new_password, user_id))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Password updated successfully"})

@app.route('/delete_account', methods=['POST'])
def delete_account():
    
    if 'user_id' in session:
        user_id = session['user_id']  # Get logged-in user ID
    

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()

            # Deleting user data from all relevant tables
            conn_users = sqlite3.connect('users.db')
            conn_predict = sqlite3.connect('predict.db')
            conn_meal = sqlite3.connect('meal.db')
            conn_exercise = sqlite3.connect('exercise.db')

            cursor_users = conn_users.cursor()
            cursor_predict = conn_predict.cursor()
            cursor_meal = conn_meal.cursor()
            cursor_exercise = conn_exercise.cursor()
            #Delete from related tables first
            cursor_predict.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
            cursor_meal.execute("DELETE FROM meal WHERE user_id = ?", (user_id,))
            cursor_exercise.execute("DELETE FROM exercise WHERE user_id = ?", (user_id,))

            # Delete from users table using id
            cursor_users.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            # Reset auto-increment sequence for all tables
            cursor_users.execute("DELETE FROM sqlite_sequence WHERE name='users'")
            cursor_predict.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
            cursor_meal.execute("DELETE FROM sqlite_sequence WHERE name='meal'")
            cursor_exercise.execute("DELETE FROM sqlite_sequence WHERE name='exercise'")
            
            # Commit changes
            conn_users.commit()
            conn_predict.commit()
            conn_meal.commit()
            conn_exercise.commit()

            # Close connections
            conn_users.close()
            conn_predict.close()
            conn_meal.close()
            conn_exercise.close()
            # Clear session data (log out user)
            session.clear()

            return redirect(url_for('register'))  # Redirect to registration page
        
        except Exception as e:
            print("Error deleting user:", e)
            return "Error deleting account"

    return redirect('/login')  # If not logged in, send to login

@app.route('/update_timezone', methods=['POST'])
def update_timezone():
    if 'user_id' not in session:
        return jsonify({"error": "User not logged in"}), 401

    user_id = session['user_id']
    data = request.get_json()
    user_timezone = data.get("timezone", "UTC")  # Default to UTC if not provided

    try:
        conn = sqlite3.connect("your_database.db")  # Use your actual DB path
        cursor = conn.cursor()

        # Update the user's timezone
        cursor.execute("UPDATE users SET timezone = ? WHERE id = ?", (user_timezone, user_id))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "timezone": user_timezone})
    except sqlite3.Error as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Contact route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        user_id = session.get('user_id')
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if not user_id:
            flash('You must be logged in to send a message!', 'danger')
            return redirect(url_for('login'))

        if not (name and email and message):
            flash('Please fill out all fields.', 'danger')
            return redirect(url_for('contact'))

        try:
            conn = sqlite3.connect('site.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO contact_messages (user_id, name, email, message) VALUES (?, ?, ?, ?)', (user_id, name, email, message))
            conn.commit()
            flash("Your message has been sent successfully!", "success")
        except sqlite3.IntegrityError as e:
            flash(f"An error occurred: {e}", "danger")
        finally:
            conn.close()

        return redirect(url_for('mainpage2'))

    return render_template('contact.html')


# Function to fetch user health details
def get_user_health_report(user_id):
    conn_users = sqlite3.connect("users.db")
    cursor_users = conn_users.cursor()
    cursor_users.execute("SELECT username, email FROM users WHERE id = ?", (user_id,))
    user = cursor_users.fetchone()
    conn_users.close()
    
    conn_predict = sqlite3.connect("predict.db")
    cursor_predict = conn_predict.cursor()
    cursor_predict.execute("SELECT age, bmi, prediction FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
    prediction = cursor_predict.fetchone()
    conn_predict.close()
    
    conn_meal = sqlite3.connect("meal.db")
    cursor_meal = conn_meal.cursor()
    cursor_meal.execute("""
            SELECT day, breakfast, lunch, dinner 
            FROM meal 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 7
        """, (user_id,))
    meals = cursor_meal.fetchall()
    
    meal_plan = [] if not meals else [
        {
            "Day": meal[0],
            "Breakfast": meal[1],
            "Lunch": meal[2],
            "Dinner": meal[3],
        }
        for meal in meals
    ]
    
    meal_status = "Meal plan available" if meals else "Meal plan not available"
    conn_meal.close()
    
    conn_exercise = sqlite3.connect("exercise.db")
    cursor_exercise = conn_exercise.cursor()
    cursor_exercise.execute("""
        SELECT workout_type, frequency, duration, exercises 
        FROM exercise 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (user_id,))
    exercise = cursor_exercise.fetchone()
    conn_exercise.close()
    
    #meal_plan_text = "\n".join([f"{meal[0]}: Breakfast - {meal[1]}, Lunch - {meal[2]}, Dinner - {meal[3]}" for meal in meals]) if meals else "No meal plan available"
    
    # Debugging Print Statements
    print(f"User Data: {user}")
    print(f"Prediction Data: {prediction}")
    #print(f"Meal Plan Data: {meal}")
    print(f"Exercise Data: {exercise}")
    
    return {
        "Name": user[0] if user else "N/A",
        "Email": user[1] if user else "N/A",
        "Age": prediction[0] if prediction else "N/A",
        "BMI": prediction[1] if prediction else "N/A",
        "Prediction": prediction[2] if prediction else "N/A",
        "Meal Plan Status": meal_status,
        #"Meal Plan": meal_plan if meal_plan else "No meal plan available",
        "Workout Type": exercise[0] if exercise else "N/A",
        "Frequency": exercise[1] if exercise else "N/A",
        "Duration": exercise[2] if exercise else "N/A",
        "Exercises": exercise[3] if exercise else "N/A"
    }



# About route
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    if user_id:
        try:
            conn = sqlite3.connect('site.db')
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS logout_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, logout_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
            cursor.execute('INSERT INTO logout_logs (user_id) VALUES (?)', (user_id,))
            conn.commit()
        finally:
            conn.close()
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
