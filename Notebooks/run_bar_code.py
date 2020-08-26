from single_cell_type_cluster import *

df = pd.read_csv('../Processing/data_list.csv')
row = df.iloc[5]

bar_code(row)