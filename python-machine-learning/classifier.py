import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("diabetes_report.html")

# Data split
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Data processing
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # Train model
# model = RandomForestClassifier(n_estimators=200, criterion="gini", random_state=100)
# model.fit(x_train, y_train)
#
# y_predict = model.predict(x_test)
# print(classification_report(y_test, y_predict))

# params = {
#     "n_estimators": [100, 200, 300],
#     "criterion": ["gini", "entropy", "log_loss"]
# }
#
# grid_search = GridSearchCV(estimator=SVC(random_state=100), param_grid=params, cv=4, scoring="recall", verbose=2)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_score_)
# print(grid_search.best_params_)
# y_predict = grid_search.predict(x_test)
# print(classification_report(y_test, y_predict))

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(predictions)






