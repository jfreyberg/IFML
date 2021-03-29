import numpy as np 

np.save('q_data/q_data.npy', np.array([np.r_[-1, np.zeros(34)]]))  # ,(16,1)))
np.save('all_data/all_data.npy', np.array([np.r_[-1, np.zeros(34)]]))  # ,(16,1)))
with open('moves.txt', 'w') as f:
    f.write('')

# np.save('thetas/theta_q.npy', np.zeros((6, 33)))


theta_alt = np.load('thetas/theta_q.npy')
# print(np.shape(theta_alt), theta_alt)
tmp = theta_alt[:, -1]
tmp = np.reshape(tmp, (6, 1))
theta_alt = np.append(theta_alt[:,:-1], np.zeros((6, 1)), axis = -1)
theta_alt = np.append(theta_alt, tmp, axis = -1)
print(np.shape(theta_alt), theta_alt)
#np.save('thetas/theta_q.npy', theta_alt)


