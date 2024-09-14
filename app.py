# app.py
import json
from flask import Flask, render_template, request, redirect, url_for, session
import pymysql
import pandas as pd
from pyNutriScore import NutriScore
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'Karthik@1234'


# MySQL Configuration
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'karthik@123'
MYSQL_DB = 'final_project'

# Connect to MySQL
conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, db=MYSQL_DB, cursorclass=pymysql.cursors.DictCursor)
# Load the CSV file once into a DataFrame
df = pd.read_csv('open_food_facts.csv')


# Route for home page
@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        search_query = request.form['search_query']
        search_results = search_names(search_query)
        search_results = json.loads(search_results)
        return render_template('home.html', search_results=search_results)
    return render_template('home.html', search_results=None)
# Function to search names in the DataFrame
def search_names(query):
    results = df[df['product_name'].str.contains(query, case=False) | df['brands'].str.contains(query, case=False)]['product_name']
    json_data = results.to_json(orient='index', force_ascii=False)
    return json_data
# Route for product page
@app.route('/product/<key>', methods=['GET'])
def product_detail(key):
    try:
        product_name = df.loc[int(key), 'product_name']
        brand = df.loc[int(key), 'brands']
        ingridients= df.loc[int(key),'ingredients_text']
        img= df.loc[int(key),'image_url']
        energy= df.loc[int(key),'energy_100g']
        fat= df.loc[int(key),'fat_100g']
        saturated_fat= df.loc[int(key),'saturated-fat_100g']
        trans_fat= df.loc[int(key),'trans-fat_100g']
        cholesterol= df.loc[int(key),'cholesterol_100g']
        carb= df.loc[int(key),'carbohydrates_100g']
        sugar= df.loc[int(key),'sugars_100g']
        fiber= df.loc[int(key),'fiber_100g']
        protein= df.loc[int(key),'proteins_100g']
        salt= df.loc[int(key),'salt_100g']
        sodium= df.loc[int(key),'sodium_100g']
        nutri_score_result,conclusions = nutri_score(key)
        # Fetch user allergy details from the database
        user_id = session.get('user_id')
        user_allergies = None
        if user_id:
            with conn.cursor() as cursor:
                cursor.execute('SELECT allergies FROM users WHERE id = %s', (user_id,))
                result = cursor.fetchone()
                if result:
                    user_allergies = result['allergies']
        user_allergies=user_allergies.split(",")
        allergens= detect_allergens(ingridients,user_allergies)
        warnings= analysis(key)
        health_conditions=None
        age=None
        gender=None
        if user_id:
            with conn.cursor() as cursor:
                cursor.execute('SELECT health_conditions,age,gender FROM users WHERE id = %s', (user_id,))
                result = cursor.fetchone()
                if result:
                    health_conditions = result['health_conditions']
                    age=result['age']
                    gender=result['gender']
        health_conditions=health_conditions.split(",")
        
        restrictions=gemini(age,gender,health_conditions)
        matching_foods_to_avoid,matching_ingredients_to_avoid, matching_foods_categories_to_avoid=final_conc(conclusions,ingridients, restrictions,warnings)
        final_conclusion=gemini2(age,gender,health_conditions,restrictions, user_allergies,matching_foods_to_avoid,ingridients,warnings).split('\n')

        product_info = {'key': key, 'product_name': product_name, 'brand': brand,'ingridients':ingridients,'image':img,'nutri_score':nutri_score_result,'conclusions':conclusions, 'allergens':allergens,'warnings':warnings,"foods_avoid":matching_foods_to_avoid,"ing_avoid":matching_ingredients_to_avoid,"cat_avoid": matching_foods_categories_to_avoid, "final_conc":final_conclusion,"energy":energy,"fat":fat,"saturated_fat":saturated_fat,"trans_fat":trans_fat,"cholestrol":cholesterol,"carbs":carb,"sugar":sugar,"fiber":fiber,"protein":protein,"salt":salt,"sodium":sodium}
        

    except KeyError:
        product_info = None
    return render_template('product_detail.html', product_info=product_info)
# Function to calculate Nutri-Score
def nutri_score(x):
    conclusions=[]
    result = NutriScore().calculate_class(
        {
            'energy': df.loc[int(x), 'energy_100g'],
            'fibers': df.loc[int(x), 'fiber_100g'],
            'fruit_percentage': df.loc[int(x), 'fruits-vegetables-nuts_100g'],
            'proteins': df.loc[int(x), 'proteins_100g'],
            'saturated_fats': df.loc[int(x), 'saturated-fat_100g'],
            'sodium': df.loc[int(x), 'sodium_100g'],
            'sugar': df.loc[int(x), 'sugars_100g'],
        },
        'solid'  # either 'solid' or 'beverage'
    )
    if result=='A':
        conclusions.extend(['healthiest choice','higher content of beneficial nutrients such as fiber, vitamins, and minerals','might include whole grains, fruits, vegetables '])
    if result=='B':
        conclusions.extend(['relatively healthy choice','offer good nutritional value','a balance of beneficial nutrients and nutrients to be limited','include certain whole grain products'])
    if result=='C':
        conclusions.extend(['moderate choice in terms of nutritional quality','might include certain processed foods,and snacks'])
    if result=='D':
        conclusions.extend(['less healthy choice','may have higher levels of nutrients to be limited, such as added sugars, saturated fats, or sodium','product should be consumed sparingly','might include sugary snacks, and certain convenience foods'])
    if result=='E':
        conclusions.extend(['least healthy choice','high levels of nutrients to be limited and low levels of beneficial nutrients','Consumption of products in this category should be limited or avoided','might include sugar, and highly processed foods'])

    return result, conclusions
# Function to detect allergies    
def detect_allergens(ingredient_list, allergens):
    detected_allergens = []

    # Convert ingredient list to lowercase for case-insensitive matching
    ingredient_list_lower = ingredient_list.lower()

    # Search for allergens in the ingredient list
    for allergen in allergens:
        if re.search(allergen, ingredient_list_lower):
            detected_allergens.append(allergen)

    return detected_allergens        
# Analysis of the product
def analysis(x):
    warnings=[]
    if df.loc[int(x),'fat_100g']>20:
        warnings.append('High fat')
    if df.loc[int(x),'saturated-fat_100g']>5:
        warnings.append('High saturated-fat')
    if df.loc[int(x),'trans-fat_100g']>0.5:
        warnings.append('High trans-fat')
    if df.loc[int(x),'cholesterol_100g']>0.3:
        warnings.append('High cholesterol')
    if df.loc[int(x),'sugars_100g']>10:
        warnings.append('High sugar')
    if df.loc[int(x),'-maltodextrins_100g']>2:
        warnings.append('contains maltodextrin')
    if df.loc[int(x),'fiber_100g']>5:
        warnings.append('High fiber')
    if df.loc[int(x),'proteins_100g']>12:
        warnings.append('High protein')
    if df.loc[int(x),'salt_100g']>1.5:
        warnings.append('High salt')
    if df.loc[int(x),'sodium_100g']>0.4:
        warnings.append('High sodium')
    if df.loc[int(x),'alcohol_100g']>0:
        warnings.append('Contains alcohol')
    if 'palm oil' in df.loc[int(x),'ingredients_text'].lower():
        warnings.append('Contains palm oil')
    return warnings
# gemini api
def gemini(age,gender,health_conditions):
    # Genrating recommendations from Gemini

    #At the command line, only need to run once to install the package via pip:

#$ pip install google-generativeai


    genai.configure(api_key="AIzaSyCXtXRmxJYyclYQ306vZ8FmduHR_79E_Sc")

    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                 safety_settings=safety_settings)

    convo = model.start_chat(history=[
    ])

    convo.send_message("what are the foods and ingridients that i need to avoid if i am suffering from"+str(health_conditions)+" and age "+str(age)+" "+gender+". give output is the form of only jason in a single line. use only foods_to_avoid and ngredients_to_avoid as keys  ")
    lr=convo.last.text
    import json

    # Sample API response (replace with your actual response)
    api_response = lr

    # Convert the JSON string to a Python dictionary
    food_restrictions = json.loads(api_response)

    return food_restrictions

def final_conc(conclusions,ing, restrictions,warnings):
    categ=conclusions+warnings
    # Sample dictionary
    avoid_dict = restrictions

    # Sample lists of ingredients and food categories
    ingredients = ing
    food_categories = categ

    # Tokenization, normalization, and removal of stopwords
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    ingredients_processed = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(" ".join(ingredients)) if word.lower() not in stop_words]
    food_categories_processed = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(" ".join(food_categories)) if word.lower() not in stop_words]

    # Matching against the dictionary
    matching_foods_to_avoid = [item for item in avoid_dict['foods_to_avoid'] if any(ingredient in item.lower() for ingredient in ingredients_processed)]
    matching_ingredients_to_avoid = [item for item in avoid_dict['ingredients_to_avoid'] if any(ingredient in item.lower() for ingredient in ingredients_processed)]

    matching_foods_categories_to_avoid = [item for item in avoid_dict['foods_to_avoid'] if any(category in item.lower() for category in food_categories_processed)]

    return matching_foods_to_avoid,matching_ingredients_to_avoid, matching_foods_categories_to_avoid

# gemini api
def gemini2(age,gender,health_conditions,restrictions, user_allergies,matching_foods_to_avoid,ingridients,warnings):
    # Genrating recommendations from Gemini

    #At the command line, only need to run once to install the package via pip:

#$ pip install google-generativeai


    genai.configure(api_key="AIzaSyCXtXRmxJYyclYQ306vZ8FmduHR_79E_Sc")

    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                 safety_settings=safety_settings)

    convo = model.start_chat(history=[
    ])

    convo.send_message("for a user with "+str(health_conditions)+"and age"+str(age)+"and gender"+gender+"you suggested the following restrictions"+str(restrictions)+" and a product contains the following ingridients"+str(ingridients)+"with"+str(warnings)+" give me a single line conclusion followed by a paragraph about why the user should consume or avoid comsuming the product")
    lr=convo.last.text
    import json

    # Sample API response (replace with your actual response)
    api_response = lr

    return api_response
    
# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if username and password match the database
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        user = cursor.fetchone()
        cursor.close()
        
        if user:
            # Store user data in session
            session['user_id'] = user['id']
            session['username'] = user['username']
            # If user exists, redirect to home page
            return redirect(url_for('home'))
        else:
            # If user does not exist, redirect to login page with error message
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

# Route for registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        height = request.form['height']
        weight = request.form['weight']
        health_conditions = request.form['health_conditions']
        allergies = request.form['allergies']
        
        # Check if username already exists in the database
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # If username already exists, redirect to registration page with error message
            return render_template('register.html', error='Username already exists')
        else:
            # If username does not exist, insert new user into the database
            cursor.execute(
                'INSERT INTO users (username, password, age, gender, height, weight, health_conditions, allergies) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
                (username, password, age, gender, height, weight, health_conditions, allergies)
            )
            conn.commit()
            cursor.close()
            # Redirect to login page after successful registration
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    user = cursor.fetchone()
    
    if request.method == 'POST':
        # Update user details in the database
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        health_conditions = request.form['health_conditions']
        allergies = request.form['allergies']
        height = request.form['height']
        weight = request.form['weight']
        
        cursor.execute(
            'UPDATE users SET username = %s, password = %s, age = %s, gender = %s, health_conditions = %s, allergies = %s, height = %s, weight = %s WHERE id = %s',
            (username, password, age, gender, health_conditions, allergies, height, weight, user_id)
        )
        conn.commit()
        cursor.close()
        
        # Update session user data if username was changed
        session['username'] = username
        
        # Redirect to home page after updating profile
        return redirect(url_for('home'))
    
    cursor.close()
    
    return render_template('profile.html', user=user)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
