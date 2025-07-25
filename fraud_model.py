import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv(r"C:\Users\user\Downloads\creditcard.csv\creditcard.csv")


print(df.head(4))

#print("\nInfo:")
#print(df.info())

#Check class distribution
#print("\nFraud class distribution:")
#print(df['Class'].value_counts())

#% of fraud cases
#fraud_rate = df['Class'].value_counts(normalize=True)[1] * 100
#print(f"\nFraud rate: {fraud_rate:.4f}%") 

#Visualising class imbalance
#sb.countplot(data=df, x='Class')
#mpl.title('Transaction Class Distribution')
#mpl.xticks([0, 1], ['Not Fraud (0)', 'Fraud (1)'])
#mpl.ylabel('Number of Transactions')
#mpl.xlabel('Class')
#mpl.show()

#Histogram of transaction amounts
#mpl.figure(figsize=(10,5))
#sb.histplot(df['Amount'], bins=50, kde=True)
#mpl.title('Distribution of Transaction Amounts')
#mpl.xlabel('Transaction Amount')
#mpl.ylabel('Frequency')
#mpl.show()

#mpl.figure(figsize=(10,5))
#sb.boxplot(x='Class', y='Amount', data=df)
#mpl.title('Transaction Amounts by Class')
#mpl.xticks([0, 1], ['Not Fraud (0)', 'Fraud (1)'])
#mpl.xlabel('Class')
#mpl.ylabel('Amount')
#mpl.yscale('log')  # Makes large outliers easier to see
#mpl.show()

X = df.drop(['Time', 'Class'], axis=1)  # Inputs
y = df['Class']                         # Output

# X = inputs, y = outputs
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Training a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print confusion matrix and classification report
#print("Confusion Matrix:")
#print(confusion_matrix(y_test, y_pred))

#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))

 # 1. Extract feature importances
importances = model.feature_importances_
features = X.columns

# 2. Put into a DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 3. Plot the top 10 important features
mpl.figure(figsize=(10, 6))
sb.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette="viridis")
mpl.title("Top 10 Most Important Features for Fraud Detection")
mpl.xlabel("Feature Importance")
mpl.ylabel("Feature")
mpl.tight_layout()
mpl.show()

#joblib.dump(model, "fraud_model.pkl")

#print(" Model saved successfully!")
