from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


emails = [
    "Win free money now",
    "Update your bank password immediately",
    "Click this link to claim reward",
    "Class starts at 9 AM",
    "Project meeting tomorrow",
    "Assignment submission is today"
]

labels = ["Phishing", "Phishing", "Phishing", "Safe", "Safe", "Safe"]


cv = CountVectorizer()
X = cv.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)


predictions = model.predict(X)


acc = accuracy_score(labels, predictions)
print("Accuracy:", acc * 100, "%")


print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

# User input
msg = input("\nEnter email text: ")

# Predict user email
test = cv.transform([msg])
result = model.predict(test)

print("Prediction:", result[0])

if result[0] == "Phishing":
    print("Warning: This email may be dangerous!")
else:
    print("This email looks safe.")
