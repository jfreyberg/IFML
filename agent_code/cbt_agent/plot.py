import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
data = np.load('thetas/theta_q.npy')
sns.heatmap(data)
plt.show()
