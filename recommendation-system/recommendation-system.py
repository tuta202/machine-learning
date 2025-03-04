import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("movie_data/movies.csv", encoding="latin-1", sep="\t", usecols=["title", "genres"])
data["genres"] = data["genres"].apply(lambda title: title.replace("|", " ").replace("-", ""))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["genres"])
tfidf_matrix_dense = pd.DataFrame((tfidf_matrix).todense(), index=data["title"], columns=vectorizer.get_feature_names_out())

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense =  pd.DataFrame(cosine_sim, index=data["title"], columns=data["title"])

# print(cosine_sim_dense)

input_movie = "Jumanji (1995)"
top_k = 20
result = cosine_sim_dense.loc[input_movie, :]
result = result.sort_values(ascending=False)[:top_k].to_frame(name="score").reset_index()

print(result)