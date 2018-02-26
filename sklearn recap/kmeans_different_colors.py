
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inlineM

df = pd.DataFrame({'x': np.random.randint(0,3, 100)})

df.loc[df.index, 'label'] = 0

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(df.x.tolist()).reshape(-1,1))

df.loc[df.index, 'label'] = kmeans.labels_

df.head()

import itertools
x = range(0,100)
fig, ax = plt.subplots(figsize=(10,10))
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'b',
                   }

label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
plt.scatter(x, y, c=label_color)
plt.show()
