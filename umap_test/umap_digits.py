import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
import umap


digits = load_digits()

# fig, ax_array = plt.subplots(20, 20)
# axes = ax_array.flatten()
# for i, ax in enumerate(axes):
#     ax.imshow(digits.images[i], cmap='gray_r')
# plt.setp(axes, xticks=[], yticks=[], frame_on=False)
# plt.tight_layout(h_pad=0.5, w_pad=0.01)

fig, ax = plt.subplots()
ax.imshow(digits.images[1658], cmap='gray_r')

reducer = umap.UMAP(random_state=42)
reducer.fit(digits.data)
embedding = reducer.transform(digits.data)

plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset')

plt.show()
