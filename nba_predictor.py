from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches  # Import difflib for fuzzy matching

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('nba_model.pkl')

# Load the cleaned data used for training (with feature stats for percentiles)
cleaned_data = pd.read_csv('/Users/joshgross/Downloads/cleaned_NBA_data.csv')

# Normalize player names in the dataset (strip whitespace, make lowercase)
cleaned_data['Name'] = cleaned_data['Name'].str.strip().str.lower()

# Define feature columns used for predictions
feature_columns = ['Age', 'DARKO', 'DRIP', 'SQ', 'FG3_PCT', 'FT_PCT', 'BPM', 'Wingspan', 'FG_PCT', 'offensiveLoad', 'TS%', 'USG%', 'BC', 'LOAD', 'PORT']

# Define target columns for the projections
target_columns = ['PTS', 'REB', 'AST', 'STL', '3P%']

# Home route for player lookup
@app.route('/')
def home():
    return render_template('lookup.html')  # Create a lookup form in the front-end

# Route to handle player lookup and display projections with percentiles
@app.route('/player', methods=['POST'])
def player_lookup():
    # Get the player's name from the form, strip extra whitespace and make it lowercase
    player_name = request.form['player_name'].strip().lower()

    # Find the closest match in the dataset (using fuzzy matching)
    player_names = cleaned_data['Name'].unique()
    matched_name = get_close_matches(player_name, [name.lower() for name in player_names], n=1, cutoff=0.6)

    # If no close match is found, return an error
    if not matched_name:
        return render_template('lookup.html', error=f'Player "{request.form["player_name"]}" not found.')

    # Use the matched name to find the player data
    player_data = cleaned_data[cleaned_data['Name'].str.lower() == matched_name[0]]

    # Get the player's features
    player_features = player_data[feature_columns].values.reshape(1, -1)

    # Make predictions for the player
    prediction = model.predict(player_features)

    # Calculate percentiles for each feature
    percentiles = {}
    for feature in feature_columns:
        feature_values = cleaned_data[feature].dropna().values  # Exclude missing values
        player_value = player_data[feature].values[0]
        # Calculate percentile for the player's feature
        percentile = np.sum(feature_values <= player_value) / len(feature_values) * 100
        percentiles[feature] = round(percentile, 2)

    # Prepare the data for display
    response = {
        'Name': matched_name[0],  # Return the matched player name
        'Prediction': {
            'Points': round(prediction[0][0], 2),
            'Rebounds': round(prediction[0][1], 2),
            'Assists': round(prediction[0][2], 2),
            'Steals': round(prediction[0][3], 2),
            '3P%': round(prediction[0][4], 2)
        },
        'Percentiles': percentiles
    }

    # Render the results in the result.html template
    return render_template('result.html', player=response)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
