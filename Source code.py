### Upload the Dataset

from google.colab import files
uploaded = files.upload()

### Load the Dataset

import pandas as pd
df = pd.read_csv("full_house_price_dataset.csv")
df.head()

### Data Exploration

df.info()
df.describe()
df.columns

### Check for Missing Values and Duplicates

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

### Visualize a Few Features

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['Price'], kde=True)
plt.title("Distribution of House Prices")
plt.show()

### Identify Target and Features

target = 'Price'
features = df.drop(columns=[target]).columns

### Convert Categorical Columns to Numerical

df.select_dtypes(include='object').columns

### One-Hot Encoding

df = pd.get_dummies(df, drop_first=True)

### Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=[target]))

### Train-Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df[target], test_size=0.2, random_state=42)

### Model Building

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

### Evaluation

### Make Predictions from New Input

from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# Save trained columns after preprocessing
trained_columns = df.drop(columns='Price').columns

# Example input (must match the format used in training)
original_input = {
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Area': 1500,
    'Garage': 1,
    'Location': 'Suburb'  # Replace with actual category used in your dataset
}

# Convert input to DataFrame
new_df = pd.DataFrame([original_input])

# One-hot encode and align columns
new_df_encoded = pd.get_dummies(new_df)
new_df_encoded = new_df_encoded.reindex(columns=trained_columns, fill_value=0)

# Scale input
scaled_input = scaler.transform(new_df_encoded)

# Predict
prediction = model.predict(scaled_input)
print("Predicted Price:", prediction[0])

### Convert to DataFrame and Encode

# Example new input (adjust keys based on your dataset)
new_input_data = {
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Area': 1500,
    'Garage': 1,
    'Location': 'Suburb'  # Replace with actual category value from your dataset
}

# Convert to DataFrame
new_df = pd.DataFrame([new_input_data])

# One-hot encode the input to match training format
new_df_encoded = pd.get_dummies(new_df)

# Ensure columns match training data
new_df_encoded = new_df_encoded.reindex(columns=trained_columns, fill_value=0)

# Scale the input using the same scaler from training
scaled_input = scaler.transform(new_df_encoded)

# Predict with the trained model
prediction = model.predict(scaled_input)
print("Predicted Price:", prediction[0])



### Predict the Final Grade (House Price)

final_prediction = model.predict(new_df)

### Deployment - Building an Interactive App

!pip install gradio
import gradio as gr

### Create a Prediction Function

def predict_price(beds, baths, area, garage):
    input_data = [[beds, baths, area, garage]]
    scaled = scaler.transform(input_data)
    return model.predict(scaled)[0]

### Create the Gradio Interface

interface = gr.Interface(fn=predict_price,
                         inputs=["number", "number", "number", "number"],
                         outputs="number",
                         title="House Price Predictor")
interface.launch()
