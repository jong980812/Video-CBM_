import csv
import pandas as pd

def get_matching_rows(df, label_value):
    matching_indices = df[df["label"] == label_value].index.tolist()
    return [i for i in matching_indices]

output_csv = 'data/video_annotation/SSV2/test_sample_num.csv'
df = pd.read_csv('data/video_annotation/SSV2/test.csv', sep=' ', header=None, names=["path", "label"]) # ucf,ssv2는 sep ' '임

label_df = pd.read_csv('data/ssv2_classes.txt', header=None, names=["name"])
label_df['id'] = label_df.index #+ 1 # ucf는 label 1부터 시작해서 +1, 나머지는 +1 지움
label_df = label_df[['id', 'name']]
label_dict = label_df.set_index("id")["name"].to_dict()

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "name", "samples_num"])  # Write the header
    print(df)
    for i, label in label_dict.items():
        matching_rows = get_matching_rows(df, i)
        writer.writerow([i, label, matching_rows])