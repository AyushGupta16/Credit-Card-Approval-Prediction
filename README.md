<H1 align ="center"> Credit Card Approval Prediction </H1>

### MODEL GENERATION
Step 1: Loading datasets
The code begins by importing essential libraries for data manipulation, visualization, and machine learning. These libraries include Pandas (pd), NumPy (np), Seaborn (sns), Matplotlib (plt), and specific modules from scikit-learn. The primary goal is to load and process two datasets: "application_record.csv" and "credit_record.csv."

Step 2: Merging datasets
In this step, the two datasets are combined into a single DataFrame (data) using the Pandas merge function. The merging is based on the 'ID' column, which is assumed to be a common identifier in both datasets. This merging allows us to have a comprehensive dataset containing both application details and credit records.

Step 3: Exploring the data
1.	*Basic Statistics*: The code uses the data.describe() function to compute and display basic statistics for the dataset. This includes statistics such as mean, standard deviation, minimum, maximum, and quartiles for numeric columns. It gives an overview of the data's central tendencies.
2.	*First Few Rows*: The data.head() function is used to display the first few rows of the merged dataset. This is useful for understanding the structure and contents of the data.

Step 4: Data Cleaning
Data cleaning is a crucial step to ensure the quality and integrity of the dataset:
-	*Missing Values Check*: The code uses data.isnull().sum() to count and display missing values in each column. This is important for identifying columns with missing data.
-	*Removing Rows with Missing Target Variable*: Rows with missing values in the 'STATUS' column are removed using data.dropna(subset=['STATUS']). This is done to ensure that only records with credit status information are retained for the modeling process.

Step 5: Defining features
A set of features that will be used as input for the machine learning model is defined. These features represent various attributes and characteristics related to the applicants, such as gender, income, education, and more. These features are stored in the 'features' list.

Step 6: Selecting features and target variable
In this step, the code selects the independent variables (features) and the dependent variable (target) to prepare the data for modeling:
- X contains the selected features from the dataset.
- y contains the target variable, which is 'STATUS,' representing the credit status.

Step 7: Converting categorical variables to numerical using one-hot encoding
Many machine learning algorithms require numerical input, so categorical variables need to be converted into a numerical format. One-hot encoding is used for this purpose. It creates binary columns for each category within the categorical variables, indicating the presence or absence of each category.


Step 8: Data Visualization
Data visualization is an essential part of data exploration. The code plots a histogram of income distribution using Seaborn and Matplotlib. This visualization provides insights into the distribution of income levels in the dataset.

Step 9: Splitting the data into training and testing sets
To assess the model's performance, the dataset is divided into training and testing subsets. The train_test_split function from scikit-learn is used for this purpose. It allocates 80% of the data for training (X_train and y_train) and 20% for testing (X_test and y_test).

Step 10: Initializing and training the Random Forest Classifier
A Random Forest Classifier is initialized with 100 decision trees. It is a machine learning model used for classification tasks. The model is then trained on the training data (X_train and y_train) to learn patterns and relationships within the data.

Step 11: Making predictions on the test set
The trained Random Forest Classifier is used to make predictions on the test set (X_test). The predicted values are stored in the 'y_pred' variable.

Step 12: Evaluating the model using classification report
To assess the model's performance, a classification report is generated using the classification_report function from scikit-learn. This report includes various metrics such as precision, recall, F1-score, and support for each class, allowing a detailed assessment of the model's ability to classify different credit statuses.

Step 13: Displaying the classification report
Finally, the classification report is displayed, providing a comprehensive summary of the model's performance. It includes accuracy and other classification metrics for different credit statuses, helping to evaluate the model's effectiveness in predicting credit outcomes.
This code represents a complete machine learning workflow, from data preprocessing to model evaluation, in the context of predicting credit statuses based on applicant and credit record data.
