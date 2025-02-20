import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.model_selection import GridSearchCV

def filter_location(location):
  match = re.search(r',\s*([A-Z]{2})$', location)
  return match.group(1) if match else location

data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data.dropna(axis=0, inplace=True)
# print(data.isna().sum())
data["location"] = data["location"].apply(filter_location)

target = "career_level"

x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

ros = SMOTEN(random_state=42, sampling_strategy={
  "bereichsleiter": 1000,
  "director_business_unit_leader": 500,
  "specialist": 500,
  "managing_director_small_medium_company": 500,
}, k_neighbors=2)
# print("before\n", y_train.value_counts())
x_train, y_train = ros.fit_resample(x_train, y_train)
# print("after\n", y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
  ("title_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
  ("location_feature", OneHotEncoder(handle_unknown="ignore"), ["location"]),
  # min_df=0.01, max_df=0.95 -> optimization
  ("description_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
  ("function_feature", OneHotEncoder(), ["function"]),
  ("industry_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

cls = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("feature_selector", SelectKBest(chi2, k=800)),
  ("model", RandomForestClassifier()),
])

# result = cls.fit_transform(x_train, y_train)
# feature_names = preprocessor.get_feature_names_out()
# selector = cls.named_steps["feature_selector"]
# selected_features = feature_names[selector.get_support()]
# print(selected_features)
# print(pd.DataFrame(result.todense()))

params = {
  "model__n_estimators": [100, 200, 300],
  "model__criterion": ["gini", "entropy", "log_loss"],
  "model__max_depth": [None, 2, 5],
}

grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring="recall_weighted", verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)
y_predicted = grid_search.predict(x_test)

# cls.fit(x_train, y_train)
# y_predicted = cls.predict(x_test)
print(classification_report(y_test, y_predicted))
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)