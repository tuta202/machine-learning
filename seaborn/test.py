import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("train.csv")

# profile = ProfileReport(data, title="Titanic", explorative=True)
# profile.to_file("titanic.html")

target = "Transported"

data.set_index("PassengerId", inplace=True)

# sns.histplot(data=data["Age"])
# sns.kdeplot(data=data["Age"])
# sns.displot(data=data, x="Age", col="Transported")
# sns.barplot(data=data, x="Transported", y="RoomService", estimator=np.mean)

# fg_instance = sns.FacetGrid(data=data, col="HomePlanet", hue="Transported")
# fg_instance.map(sns.scatterplot, "RoomService", "Spa", s=50, edgecolor="b", alpha=0.5)
# fg_instance.add_legend()

sns.scatterplot(data=data, x="RoomService", y="Spa", hue="HomePlanet")

# sns.pairplot(data=data, hue="HomePlanet")

plt.show()