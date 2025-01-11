import pandas as pd

txt_file_path = 'data/ucf101_classes.txt'
csv_file_path = 'data/ucf101_classes.csv'

with open(txt_file_path, 'r') as file:
    lines = [line.strip() for line in file if line.strip()]

data = {'id': range(1, len(lines) + 1), 'name': lines}
df = pd.DataFrame(data)

df.to_csv(csv_file_path, index=False)