from utils import *
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('../Processing/data_list.csv')
row = df.iloc[5]
save_root = row['save_dir']+'/'
print(row['dat_dir'])

A_center = np.load(save_root+'cell_center.npy')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A_center[:, 1], A_center[:, 2], A_center[:, 0])
plt.title('cell centers')
plt.show()
