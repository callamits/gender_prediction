# -*- coding: utf-8 -*-
"""

Code for gender_prediction. 

Its a NON-WORKING code with examples of code snippets which you can use to build your own gender prediction model


"""

from warnings import warn
import pandas as pd # data processing
data = pd.read_csv('gender_data.csv',encoding='ISO 8859-1')

print(data)

#data = pd.read_csv(path,  encoding='latin-1')         # Reading the .csv file using pandas
data_size = len(data)

data.head()

# Percentage of each category in our dataset
data['gender'].value_counts()/ data_size * 100


print("data count :-",len(data))

# Remove duplicates
duplicate_text_data = data[data.duplicated()]
print("duplicate data count :-",len(duplicate_text_data))

duplicate_text_data

index_of_duplicate_data = duplicate_text_data.index
index_of_duplicate_data[:5]

unique_data = data.drop(index_of_duplicate_data)
print("unique data count :-", len(unique_data))

print("Duplicate Article Diff")
(data['gender'].value_counts()- unique_data['gender'].value_counts())

# Vectorize using TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True,min_df = 5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')

features = tfidf.fit_transform(unique_data.content).toarray()

# Associate Category names with numerical index and save it in new column category_id
unique_data['category_id'] = unique_data['gender'].factorize()[0]

#View first 10 entries of category_id, as a sanity check
unique_data['category_id'][0:10]

unique_data.head(9)

category_id_df = unique_data[['gender', 'category_id']].drop_duplicates().sort_values('category_id')

category_id_df

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'gender']].values)

id_to_category

unique_data.sample(2, random_state=0)

labels = unique_data.category_id

print(f"X features : {features[0]} and y : {labels[0]}\n")
print(f"X features : {features[1]} and y : {labels[1]}\n")
print(f"X features : {features[2]} and y : {labels[2]}\n")
print(f"Shape of X: {features.shape} and shape of y : {labels.shape}")

# Verify to test which models provides maximum accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initializing a list of models 

models = [
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

entries = []
accuracy = []

for model in models:
    model_name = model.__class__.__name__
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, unique_data.index, \
                                                                                     test_size=0.20, random_state=0)
    model.fit(X_train, y_train)
    accuracy.append(accuracy_score(y_test, model.predict(X_test)))
    entries.append((model_name))

model_acc = pd.DataFrame(list(zip(entries, accuracy)), columns = ['model_name', 'accuracy'])

model_acc

from sklearn.model_selection import train_test_split

# Accuracy check for RandomForestClassifier you can similarly check other models
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
#model = MultinomialNB()
#model = LogisticRegression(random_state=0)


#Split Data 
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, unique_data.index, \
                                                                                 test_size=0.50, random_state=0)

#Train Algorithm
model.fit(X_train, y_train)

# Make Predictions
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

import pickle
filename = 'finalized_model_gender'
pickle.dump(model, open(filename, 'wb'))
# Dump the file
pickle.dump(tfidf, open("tfidftransformer_gender", "wb"))
pickle.dump(id_to_category, open("id_dict", "wb"))