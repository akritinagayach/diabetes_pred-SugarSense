import sqlite3
from werkzeug.security import generate_password_hash


def create_database():
    conn = sqlite3.connect("users.db")  # Create or connect to a database
    cursor = conn.cursor()

    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            timezone TEXT DEFAULT 'UTC',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Users table created successfully!")

def update_user_details(user_id, new_username, new_email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET username=?, email=? WHERE id=?", (new_username, new_email, user_id))
    conn.commit()
    conn.close()

def update_password(user_id, new_password):
    hashed_password = generate_password_hash(new_password)
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM users")
    users = cursor.fetchall()

    for user_id, password in users:
        if not password.startswith("pbkdf2:sha256"):  # If password is not hashed
            hashed_password = generate_password_hash(password)
            cursor.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
    conn.commit()
    conn.close()
    
    

    

# Create predictions database and table
def create_predictions_database():
    conn = sqlite3.connect("predict.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            genhlth INTEGER,
            highbp INTEGER,
            bmi REAL,
            diffwalk INTEGER,
            highchol INTEGER,
            age INTEGER,
            heartdisease INTEGER,
            physhealth INTEGER,
            stroke INTEGER,
            menthealth INTEGER,
            cholcheck INTEGER,
            smoker INTEGER,
            prediction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Predictions table created successfully!")
    


    # Create contact messages table in the site database
def create_contact_messages_table():
    conn = sqlite3.connect('site.db')
    cursor = conn.cursor()
    # Check if 'user_id' column exists
    #cursor.execute("PRAGMA table_info(contact_messages)")
    #columns = [column[1] for column in cursor.fetchall()]

    #if 'user_id' not in columns:
        #cursor.execute('ALTER TABLE contact_messages ADD COLUMN user_id INTEGER')

    # Create contact_messages table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')

    conn.commit()
    conn.close()
    print("✅ Contact messages table created successfully!")
    
    
    # Create feedback table
def create_feedback_table():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("PRAGMA table_info(feedback)")
    columns = [column[1] for column in c.fetchall()]
    
    

    # Create feedback table if it doesn't exist (just in case)
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    feedback TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
    
    
    conn.commit()
    conn.close()
    print("Feedback table created successfully")
    
# Create chatbot table
def create_chatbot_table():
    conn = sqlite3.connect('chatbot.db')
    cursor = conn.cursor()

    # Create chatbot table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS chatbot_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')

    conn.commit()
    conn.close()
    print("✅ Chatbot history table created successfully!")

# Create meal plan table
def create_meal_plan_table():
    conn = sqlite3.connect('meal.db')
    cursor = conn.cursor()
    #cursor.execute("DROP TABLE IF EXISTS meal")
    cursor.execute('''CREATE TABLE IF NOT EXISTS meal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date DATE DEFAULT CURRENT_DATE,
    day TEXT CHECK(day IN ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')) NOT NULL,
    breakfast TEXT NOT NULL,
    lunch TEXT NOT NULL,
    dinner TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, day)
)''')

    conn.commit()
    conn.close()
    print("✅ Meal plan table created successfully!")


# Create exercise plan table
def create_exercise_plan_table():
    conn = sqlite3.connect('exercise.db')
    cursor = conn.cursor()
    
    #cursor.execute("DROP TABLE IF EXISTS exercise")

    cursor.execute('''CREATE TABLE IF NOT EXISTS exercise (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            workout_type TEXT NOT NULL,
            frequency TEXT NOT NULL,
            duration TEXT NOT NULL,
            exercises TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')

    conn.commit()
    conn.close()
    print("✅ Exercise plan table created successfully!")




if __name__ == "__main__":
    
    create_database()
    create_predictions_database()
    create_contact_messages_table()
    create_feedback_table()
    create_chatbot_table()
    create_meal_plan_table()
    create_exercise_plan_table()
    
    
   
