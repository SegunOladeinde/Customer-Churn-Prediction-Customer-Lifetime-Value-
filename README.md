# 🎯 ChurnGuard AI: Customer Churn Prediction & CLV Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segscustomerchurnpredict.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A friendly tool that helps businesses predict which customers might quit their service and figure out which customers are most valuable to keep.**

*Think of it like a crystal ball for businesses - it helps them see into the future and make smart decisions about keeping customers happy!*

## 📋 What You'll Find in This Guide

- [🎯 What is This Project?](#-what-is-this-project)
- [✨ What Can It Do?](#-what-can-it-do)
- [🏗️ How Does It Work?](#️-how-does-it-work)
- [📁 Project Files Explained](#-project-files-explained)
- [🛠️ How to Get It Running](#️-how-to-get-it-running)
- [🚀 How to Use It](#-how-to-use-it)
- [🔧 Making Changes](#-making-changes)
- [🤝 Want to Help Improve It?](#-want-to-help-improve-it)
- [📄 Legal Stuff](#-legal-stuff)

## 🎯 What is This Project?

**The Big Problem**: Many companies lose money when customers stop using their service (we call this "churn"). It's like when people cancel their Netflix subscription or stop going to a gym.

**Our Solution**: We built a smart computer program that can:
1. **Guess who might quit** - Like having a friend who can tell when you're about to give up on something
2. **Figure out customer value** - Calculate how much money each customer will bring to the business over time
3. **Show why it makes these guesses** - Explain its thinking like a teacher showing work on a math problem

**Who is this for?** Business owners, managers, and anyone who wants to keep customers happy and save money.

## ✨ What Can It Do?

### 🔮 Predict Who Might Quit (Churn Prediction)
- **Uses 3 different smart programs** working together (like having 3 friends give you advice)
- **Instant answers** - Type in customer info and get a "quit risk" score right away
- **Smart clues** - Creates helpful information from basic customer data
- **Flexible risk levels** - You decide what counts as "high risk"

### 💰 Customer Value Calculator (CLV Analysis)
- **Customer groups** - Sorts customers into 4 groups based on their value
- **Smart alerts** - Shows which valuable customers are at risk of leaving
- **Pretty charts** - Makes graphs to help you understand the data
- **Money talk** - Shows how much money is at risk

### 📊 Performance Checker
- **Score cards** - Shows how well each smart program is doing
- **Comparison charts** - Like report cards comparing different students
- **Important factors** - Shows what matters most in decisions
- **Explanation pictures** - Visual guides showing why decisions are made

### 🎨 Easy-to-Use Interface
- **Clean design** - Looks professional and is easy to read
- **Three sections** - Organized like tabs in a notebook
- **Custom styling** - Pretty colors and layouts
- **Works on different screens** - Looks good on phones, tablets, and computers

## 🏗️ How Does It Work?

Imagine you're teaching a computer to recognize cats. You show it thousands of cat pictures, and eventually it learns what makes a cat a cat. That's basically what we did here!

```
📂 Project Files:
├── app.py                 # The main website (like the front door)
├── src/                   # Helper programs (like tools in a toolbox)
│   ├── data_prep.py       # Data cleaner and organizer
│   ├── train_models.py    # Teacher that trains the smart programs
│   ├── predict.py         # Fortune teller for new customers
│   ├── clv_analysis.py    # Money calculator
│   └── interpretability.py # Explainer (like showing your work in math)
├── data/                  # Information storage
│   ├── raw/               # Raw customer data (like uncooked ingredients)
│   └── processed/         # Clean, ready-to-use data (cooked meal)
├── models/                # Trained smart programs (like trained dogs)
├── figures/               # Charts and graphs (like drawings)
├── assets/                # Pretty styling (like decorations)
└── requirements.txt       # Shopping list of needed tools
```

### The Journey of Data
1. **Get Data Ready** → Clean up messy info → Create smart clues → Organize everything
2. **Train Smart Programs** → Teach 3 different programs → Test them → Save the best ones
3. **Calculate Money Value** → Figure out customer worth → Group them → Make charts
4. **Create Explanations** → Show why decisions are made → Make it understandable
5. **Build Website** → Load smart programs → Make predictions → Show results

## 📁 Project Files Explained

### Main Helper Programs (in the `src/` folder)

#### `src/data_prep.py` - The Data Chef 🍳
This program is like a chef preparing ingredients:
- Gets customer data from a big spreadsheet (7,043 customers!)
- Fixes missing information (like filling in blanks)
- Creates smart clues from basic info:
  - `tenure_bucket`: Groups customers by how long they've been with us (0-6 months, 6-12 months, etc.)
  - `services_count`: Counts how many services each customer uses
  - `monthly_to_total_ratio`: Compares monthly bills to total spending
- Changes text answers to numbers (like changing "Yes/No" to 1/0)
- Splits data into practice and test groups (like dividing a pizza fairly)

#### `src/train_models.py` - The Teacher 👩‍🏫
This teaches three different smart programs:
- **Logistic Regression** - Simple, reliable (like a good student)
- **Random Forest** - Smart group decision maker (like a team of experts)
- **XGBoost** - Super smart and fast (like a genius)
- Tests them with practice questions
- Saves the trained programs for later use

#### `src/predict.py` - The Fortune Teller 🔮
Makes predictions for new customers:
- Loads the trained smart programs
- Takes customer info and prepares it
- Asks each program "Will this customer quit?"
- Gives a percentage chance (like "70% chance they'll stay")

#### `src/clv_analysis.py` - The Money Calculator 💰
Figures out how valuable customers are:
- Calculates lifetime value using past data
- Groups customers by value (top 25%, bottom 25%, etc.)
- Makes charts showing money patterns
- Compares quit risk with customer value

#### `src/interpretability.py` - The Explainer 📚
Shows why the smart programs make certain guesses:
- Uses special tools to understand decisions
- Shows which customer details matter most
- Makes charts explaining individual predictions
- Creates summary pictures of patterns

### Important Data Files

#### `data/processed/` - Ready-to-Use Information
- `train.csv`, `val.csv`, `test.csv`: Practice and test data splits
- `encoding_mapping.json`: Translation guide for text-to-numbers
- `clv_insights.txt`: Money value findings and tips

#### `models/` - Trained Smart Programs
- Three trained programs saved as files
- Tools for preparing data the same way
- Explanation helpers
- Score comparison sheets

#### `figures/` - Charts and Pictures
- Money value distribution charts
- Performance comparison graphs
- Explanation visualizations
- Feature importance pictures

## 🛠️ How to Get It Running

### What You Need First
- **Python 3.11 or newer** (the computer language we use)
- **pip** (a tool that installs Python helpers)

### Step-by-Step Setup

1. **Get the project files** (like downloading a game)
   ```bash
   git clone https://github.com/SegunOladeinde/Customer-Churn-Prediction-Customer-Lifetime-Value-.git
   cd Customer-Churn-Prediction-Customer-Lifetime-Value-
   ```

2. **Create a clean workspace** (like having your own room)
   ```bash
   python -m venv venv
   # On Windows, type:
   venv\Scripts\activate
   # On Mac/Linux, type:
   source venv/bin/activate
   ```

3. **Install the tools** (like gathering ingredients for cooking)
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the website** (like opening a store)
   ```bash
   streamlit run app.py
   ```

The website will open in your internet browser at `http://localhost:8501`!

## 🚀 How to Use It

### The Website Interface

Our website has three main sections (like chapters in a book):

#### 1. Make Predictions 📝
- Fill out the customer information form on the side
- See instant "quit risk" scores
- Compare what each smart program thinks
- Get explanations for why they made that guess

#### 2. Check Performance 📊
- See how well each program is doing (like grades)
- Look at comparison charts
- Find out which customer details matter most
- Explore explanation pictures

#### 3. Money Analysis 💵
- See how customer values are spread out
- Check quit risk by value groups
- Get business tips and advice
- Look at customer group charts

### Using the Code Directly

If you want to use the smart programs in your own code:

```python
# Import the helper programs
from src.predict import load_models, prepare_input_data

# Load the trained smart programs
models = load_models()

# Describe a customer
customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,  # 0 = No, 1 = Yes
    'Partner': 'Yes',
    'tenure': 24,        # months they've been a customer
    'MonthlyCharges': 65.0,  # monthly bill
    # ... more details
}

# Get ready to make a prediction
features = prepare_input_data(customer)

# Ask the first program for a guess
prediction = models[0].predict_proba(features)
quit_chance = prediction[0][1]  # Number between 0 and 1
```

## 🔧 Making Changes

### Basic Settings
You don't need to change anything to use it normally.

### Training New Smart Programs
If you want to retrain the programs (like teaching new tricks):

```bash
cd src
python train_models.py
python clv_analysis.py
python interpretability.py
```

### Using Different Data
To use your own customer data:
1. Put your data file in the `data/raw/` folder
2. Change the `src/data_prep.py` file to match your data
3. Retrain the smart programs

## 🤝 Want to Help Improve It?

We love when people help make things better! Here's how:

1. **Make your own copy** of the project (called "forking")
2. **Create a new branch** for your changes: `git checkout -b my-awesome-improvement`
3. **Save your changes**: `git commit -m 'Added something cool'`
4. **Share your changes**: `git push origin my-awesome-improvement`
5. **Ask to add it** by making a "Pull Request"

### Tips for Contributors
- Write clean, easy-to-read code
- Test your changes to make sure they work
- Update this guide if you add new features
- Make sure all tests pass before sharing

## 📄 Legal Stuff

This project uses the MIT License - it's free to use and modify, but please give credit!

---

**Made with ❤️ to help businesses keep customers happy**

*Created by Segun Oladeinde | Data Scientist & ML Engineer*