# Import of package
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# get the data
df= pd.read_csv('https://raw.githubusercontent.com/liadmagen/Modeling_course/main/data/spam_or_not_spam.csv')
print(df.head())
print(len(df))
# removing the NA values
df=df.dropna()
print(len(df))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.email, df.label, test_size=0.2, random_state=42)

# Create the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ComplementNB(alpha=1e-2))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# accuracy of 99 %