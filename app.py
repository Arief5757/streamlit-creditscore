import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Transform categorical variables to numeric values
data_transformed = data.copy()
label_encoders = {}
for column in ['Gender', 'Education', 'Marital Status', 'Home Ownership', 'Credit Score']:
    le = LabelEncoder()
    data_transformed[column] = le.fit_transform(data_transformed[column])
    label_encoders[column] = le

# Define Independent and Dependent Variables
X = data_transformed[['Age', 'Gender', 'Income', 'Marital Status', 'Home Ownership']]
y = data_transformed['Credit Score']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Streamlit app
st.title("Klasifikasi kredit skor metode KNN")

# Input fields
st.sidebar.header("Kolom input data")
def user_input_features():
    age = st.sidebar.slider('Umur', 18, 100, 30)
    gender = st.sidebar.selectbox('Jenis kelamin', label_encoders['Gender'].classes_)
    income = st.sidebar.slider('Pendapatan', 10000, 1000000, 50000)
    marital_status = st.sidebar.selectbox('Status', label_encoders['Marital Status'].classes_)
    home_ownership = st.sidebar.selectbox('Kepemilikan tanah', label_encoders['Home Ownership'].classes_)

    data = {'Age': age,
            'Gender': gender,
            'Income': income,
            'Marital Status': marital_status,
            'Home Ownership': home_ownership}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Transform user input features
for column in ['Gender', 'Marital Status', 'Home Ownership']:
    input_df[column] = label_encoders[column].transform(input_df[column])

# Predict the credit score
prediction = knn.predict(input_df)
prediction_proba = knn.predict_proba(input_df)

st.subheader('Data input')
st.write(input_df)

st.subheader('Prediksi klasifikasi')
credit_score = label_encoders['Credit Score'].inverse_transform(prediction)[0]
st.write(f"Prediksi standar kekayaan : {credit_score}")

st.subheader('Probabilitas klasifikasi')
prediction_proba_df = pd.DataFrame(prediction_proba, columns=label_encoders['Credit Score'].classes_)
st.write(prediction_proba_df)

# Display Model Evaluation
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

st.subheader('Evaluasi model')
st.write(f"Akurasi : {accuracy}")
st.write(f"Presisi : {precision}")
st.write(f"Skor recall : {recall}")
st.write(f"Skor F1 : {f1}")
st.write("Confusion Matrix :")
st.write(conf_matrix)

# Display Correlogram (Heatmap)
st.subheader('Korelasi (Heatmap)')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data_transformed.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
