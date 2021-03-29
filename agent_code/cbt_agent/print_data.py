import numpy as np
import matplotlib.pyplot as plt

file = 'theta_q'
theta = np.load('thetas/{}.npy'.format(file))
print('Theta hat shape', np.shape(theta), 'und lautet:')
print(theta)
plt.figure(figsize = (18, 5))
plt.imshow(theta, cmap = 'gray_r')
plt.colorbar(shrink = 0.6)
#plt.savefig('{}.png'.format(file), format= 'png')
plt.show()
