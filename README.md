Calories Burnt Prediction
This project uses a machine learning model to predict the number of calories burnt during physical activities based on input features such as age, gender, height, weight, duration, and heart rate. The model was trained using the CatBoost Regressor, which is optimized for performance and handles categorical data effectively.

Table of Contents
Project Overview
Data Collection
Data Preprocessing
Exploratory Data Analysis
Model Training
Evaluation
Results
How to Use
Acknowledgements
Project Overview
This project aims to develop a predictive model for calorie expenditure, using data about physical activity and individual attributes. The dataset was combined from two sources—exercise data and calorie data—and processed to generate meaningful insights and predictions.

The CatBoost Regressor algorithm was used for training due to its robustness with tabular datasets and its ability to handle categorical variables without extensive preprocessing.

Data Collection
The project utilizes two CSV datasets:

Exercise Data: Contains features like age, gender, height, weight, duration, and heart rate.
Calories Data: Includes the calories burnt for corresponding exercise data.
These datasets were merged to form a comprehensive dataset for training and evaluation.

Data Preprocessing
Steps:
Combined the exercise and calorie datasets.
Replaced categorical data (e.g., gender) with numerical labels.
Checked for missing values and cleaned the data.
Separated the features and target variable for model training.
Exploratory Data Analysis
Key insights and visualizations:

Distribution plots for age, height, weight, and duration.
Correlation heatmap for numerical features to understand relationships.
Count plot for gender distribution.
Model Training
Algorithm: CatBoost Regressor
CatBoost was selected for its ability to handle categorical variables and provide excellent results without extensive hyperparameter tuning.
Dataset Splitting
Training set: 80%
Test set: 20%
Evaluation
The model was evaluated using the Mean Absolute Error (MAE) metric. The following results were achieved:

MAE: A low error value, demonstrating the model's accuracy in predicting calories burnt.
Results
The model successfully predicts calorie expenditure based on the provided input features. It generalizes well to unseen data, as evident from its evaluation metrics.

How to Use
Clone this repository:
bash command:
git clone <repository-url>
Install the required dependencies:
bash
pip install -r requirements.txt
Run the notebook or script to train the model and make predictions.
Acknowledgements
Datasets: Exercise and Calorie Data
Tools: Python, Pandas, Seaborn, Matplotlib, Scikit-learn, CatBoost
Visualization: Seaborn, Matplotlib
