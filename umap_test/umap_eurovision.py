import pandas as pd
import matplotlib.pylab as plt
import umap

df = pd.read_excel("eurovision.ods")

# # 2-D
# reducer = umap.UMAP(n_neighbors=4)
# embedding = reducer.fit_transform(df.values.T)
#
# fig, ax = plt.subplots()
# ax.scatter(embedding[:, 0], embedding[:, 1])
# ax.set_aspect('equal', 'datalim')
# ax.set_title('UMAP projection of eurovision scores')
#
# labels = []
# for i, name in enumerate(df.keys()):
#     labels.append(ax.text(embedding[i, 0] + 0.05, embedding[i, 1] + 0.05, str(name)))
#
# ax.spines[['right', 'top']].set_visible(False)

# 3-D
reducer = umap.UMAP(n_neighbors=4, n_components=3)
embedding = reducer.fit_transform(df.values.T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])

labels = []
for i, name in enumerate(df.keys()):
    labels.append(ax.text(embedding[i, 0] + 0.05, embedding[i, 1] + 0.05, embedding[i, 2] + 0.05, str(name)))

plt.show()
