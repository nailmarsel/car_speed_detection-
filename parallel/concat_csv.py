import pandas as pd
import os

root_catalog = r'C:\Users\user\PycharmProjects\particle_detection\utils'

csvs = []
print(os.listdir(root_catalog))
for file in os.listdir(root_catalog):
    if file[file.find('.')+1:] == 'csv':
        csvs.append(file)

result_df = pd.concat([pd.read_csv(csv, sep=';') for csv in csvs])
print(result_df)
result_df.set_index('file_name').to_csv('result.csv', sep=';')