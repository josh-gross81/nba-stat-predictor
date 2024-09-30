import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # Import joblib to save the model

# Load both datasets
cleaned_data = pd.read_csv('/Users/joshgross/Downloads/cleaned_NBA_data.csv')
additional_data = pd.read_csv('/Users/joshgross/Downloads/cleaned_additional_NBA_data.csv')

# Reintroduce player names from the original dataset
cleaned_data['Name'] = cleaned_data['Name'].str.strip()
additional_data['Name'] = additional_data['Name'].str.strip()

# Merge the datasets on the 'Name' column
merged_data = pd.merge(cleaned_data, additional_data[['Name', 'PTS', 'REB', 'AST', 'STL', '3:00 PM']], on='Name', how='inner')

# Rename '3:00 PM' to '3P%' for clarity
merged_data.rename(columns={'3:00 PM': '3P%'}, inplace=True)

# Ensure 3P% values are in the range 0-1 by dividing by 100 if necessary
# (Assuming some values are percentages and not decimals.)

# Convert 'Age' and other numeric columns to appropriate data types
merged_data['Age'] = pd.to_numeric(merged_data['Age'], errors='coerce')

numeric_columns = ['DARKO', 'DRIP', 'SQ', 'FG3_PCT', 'FT_PCT', 'BPM', 'Wingspan', 'FG_PCT', 'offensiveLoad', 'TS%', 'USG%', 'BC', 'LOAD', 'PORT']
merged_data[numeric_columns] = merged_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Define feature columns and target columns
feature_columns = ['Age', 'DARKO', 'DRIP', 'SQ', 'FG3_PCT', 'FT_PCT', 'BPM', 'Wingspan', 'FG_PCT', 'offensiveLoad', 'TS%', 'USG%', 'BC', 'LOAD', 'PORT']
target_columns = ['PTS', 'REB', 'AST', 'STL', '3P%']

# Drop rows with missing values to ensure clean data
merged_data_clean = merged_data.dropna(subset=feature_columns + target_columns)

# Define X (features) and y (targets)
X = merged_data_clean[feature_columns]
y = merged_data_clean[target_columns]

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a multi-output linear regression model
model = MultiOutputRegressor(LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Train the model on the entire dataset (for final predictions)
model = MultiOutputRegressor(LinearRegression())
model.fit(X, y)

# Make predictions for all players
y_pred = model.predict(X)

# Add player names back to the predictions
player_names = merged_data_clean['Name'].values
predicted_stats = pd.DataFrame(y_pred, columns=target_columns)

# Ensure the number of player names matches the number of predicted stats
if len(player_names) == len(predicted_stats):
    predicted_stats.insert(0, 'Name', player_names)
else:
    print(f"Warning: Length mismatch between player names ({len(player_names)}) and predicted stats ({len(predicted_stats)})")
    
# Sort by Points Per Game (PTS)
predicted_stats_sorted = predicted_stats.sort_values(by='PTS', ascending=False)

# Convert the DataFrame to HTML and add a search bar with JavaScript
html_output = predicted_stats_sorted.to_html(index=False)

# Create HTML file with search functionality
output_html_path = '/Users/joshgross/Downloads/nba_predictions_with_search.html'

with open(output_html_path, 'w') as f:
    f.write("""
    <html>
    <head>
        <title>NBA Player Predictions</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 40px;}
            table {border-collapse: collapse; width: 100%;}
            th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;}
            th {background-color: #4CAF50; color: white;}
            #searchInput {margin-bottom: 20px; padding: 10px; width: 100%;}
        </style>
        <script>
            function searchTable() {
                var input, filter, table, tr, td, i, txtValue;
                input = document.getElementById("searchInput");
                filter = input.value.toUpperCase();
                table = document.getElementById("predictionsTable");
                tr = table.getElementsByTagName("tr");

                for (i = 1; i < tr.length; i++) {
                    td = tr[i].getElementsByTagName("td")[0];
                    if (td) {
                        txtValue = td.textContent || td.innerText;
                        if (txtValue.toUpperCase().indexOf(filter) > -1) {
                            tr[i].style.display = "";
                        } else {
                            tr[i].style.display = "none";
                        }
                    }
                }
            }
        </script>
    </head>
    <body>
        <h1>NBA Player Predictions</h1>
        <input type="text" id="searchInput" onkeyup="searchTable()" placeholder="Search for players..">
        <table id="predictionsTable">
        """ + html_output + """
        </table>
    </body>
    </html>
    """)

print(f"Predicted stats for all players with search functionality have been exported to {output_html_path}")
