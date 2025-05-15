# **SugarSense: Diabetes Prediction System**

## **Machine Learning Application for Diabetes Risk Prediction**

SugarSense is an intelligent web application that empowers users to assess their diabetes risk (diabetic, prediabetic, or non-diabetic) and receive culturally tailored Indian meal and fitness recommendations. With the integration of AI-driven chat and voice interfaces, SugarSense provides a holistic digital health experience.

## **Overview**

In this project, we:

- Predict diabetes status using a machine learning model trained on clinical and behavioral data.
- Generate weekly Indian meal plans and fitness routines tailored to user profiles.
- Integrate text-based and voice-based AI assistants for seamless user interaction.
- Enable user account management and personalized settings.
- Provide downloadable health reports summarizing prediction, diet, and fitness plans.

## **Key Features**

**Diabetes Prediction**

- Accepts user inputs: gender, age, BMI, cholesterol, blood pressure, physical/mental health indicators, and more.
- Predicts whether a user is Diabetic, Prediabetic, or Non-Diabetic.
- Uses pre-trained machine learning models for accurate classification.
- Prediction results stored in predict.db.

**Personalized Indian Meal Planning**

- Tailored to age, BMI, gender, and dietary preference (veg/non-veg).
- Weekly plan includes breakfast, lunch, and dinner for each day.
- Incorporates user prediction result into dietary customization.
- Plans stored in meal.db.

**Exercise Recommendations**

- Personalized workout plan based on user's age, BMI, and gender.
- Categories include workout type, duration, and frequency.
- Stored in exercise.db.

**Chatbot Support**

- AI-based text assistant to answer questions, guide users, and provide health insights.
- Integrates with backend services and database queries.
- Built using Chatbase API.

**Vihaan – Voice Assistant**

- Built using ElevenLabs API.
- Accepts voice commands to run predictions, generate plans, or answer queries.
- Enhances accessibility and user experience.

**PDF Health Report**

- Combines all user data into a comprehensive downloadable report.
- Includes prediction result, personalized meal plan, and exercise plan.
- Generated dynamically and downloadable via the user dashboard.

**User Dashboard & Settings**

- Users can:
  - Edit username/email
  - Change password (securely hashed)
  - View the history of predictions, meals, and workouts
  - Download report.
  - Provide feedback or contact support

## **Objectives**

- Enable early prediction and awareness of diabetes conditions.
- Deliver proactive healthcare solutions for diabetes management.
- Empower users with actionable health plans tailored to Indian culture.
- Utilize AI (chat + voice) for accessible and intelligent support.
- Provide transparency and interpretability through downloadable reports.

## **Methodology**

1. **User Data Collection**
    - Inputs collected during registration and health prediction.
2. **ML-Based Prediction**
    - Model trained on structured health data to classify users as diabetic, prediabetic, or non-diabetic.
3. **Meal & Exercise Recommendation System**
    - Meal plans generated using API + rules based on demographics and health status.
    - Exercise plans queried from the exercise database.
4. **Chatbot and Voice AI**
    - Chatbot responds to natural language queries.
    - Voice assistant executes key functions using speech commands.
5. **Report Generator**
    - Collates prediction, diet, and fitness data into a well-formatted PDF.

## **Database Schema**

**users.db**

  - Stores: id, username, email, password, timezone, created_at

**predict.db**

  - Stores prediction input features and result:
  - user_id, age, bmi, highbp, cholcheck, smoker, etc.
  - prediction result

**meal.db**

  - Meal plan per user per day:
  - user_id, day, breakfast, lunch, dinner, timestamp

**exercise.db**

  - Personalized fitness plans:
  - user_id, workout_type, frequency, duration, exercises, timestamp

**feedback.db**

  - User-submitted feedback:
  - user_id, feedback

**site.db**

  - Contact messages from users:
  - user_id, name, email, message, timestamp

## **Project Structure**

SugarSense/

├── app.py           # Main Flask application

├── static/          # Static assets (CSS, JS, Images)

├── templates/       # HTML templates

├── database/        # DB initialization and schema setup

├── users.db         # User information

├── predict.db       #Diabetes predictions

├── meal.db          # Meal plans

├── exercise.db      # Fitness plans

├── feedback.db      #Feedback

├── site.db          #Contact us

├── requirements.txt # Python dependencies

└── README.md        # Project documentation

## **Getting Started**

**Prerequisites**

- Python 3.8 +
- Install dependencies:

pip install -r requirements.txt

**Usage**

1. **Clone the repository**

      git clone <https://github.com/akritinagayach/diabetes_pred-SugarSense.git>

1. **Navigate to the project directory**

      cd SugarSense

1. **Run the Flask app**

      python app.py

## **Contributing**

We welcome contributions! Please follow the steps below:

1. Fork the repository.
   
2. Create a new branch:

      git checkout -b feature-branch

3. Commit your changes:

      git commit -m "Add new feature"

4. Push to your branch:

      git push origin feature-branch

5. Submit a pull request.

## **License**

This project is licensed under the MIT License.

## **Acknowledgments**

We acknowledge the invaluable contribution of clinical datasets and the insights from related literature that informed this study.

- chatbase for AI Chatbot
- [ElevenLabs](https://www.elevenlabs.io/) for voice assistant
- [SQLite](https://www.sqlite.org/index.html) for database support
- All open-source contributors

## **Contact**

- Akriti Nagayach
- GitHub: [akritinagayach (Akriti Nagayach)](https://github.com/akritinagayach)
