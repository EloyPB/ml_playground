import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap


if os.path.isfile("penguins.csv"):
    print("Loading penguins.csv...")
    penguins = pd.read_csv("penguins.csv")
else:
    print("Downloading penguins.csv...")
    penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
    penguins.to_csv("penguins.csv", index=False)

penguins = penguins.dropna()

# sns.pairplot(penguins.drop("year", axis=1), hue='species')

penguin_data = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].values
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_penguin_data)

plt.scatter(embedding[:, 0], embedding[:, 1],
            c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie": 0, "Gentoo": 1, "Chinstrap": 2})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset')

plt.show()
