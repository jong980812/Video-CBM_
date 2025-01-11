# import pandas as pd

# mimetics_df = pd.read_csv("data/video_annotation/mimetics50/mimetics_v1.0.csv")
# kinetics_classes_df = pd.read_csv("data/kinetics400_classes.csv")

# label_to_id = dict(zip(kinetics_classes_df['name'], kinetics_classes_df['id']))

# output_rows = []

# for _, row in mimetics_df.iterrows():
#     video_path = f"/local_datasets/mimetics/{row['youtube_id']}_{row['time_start']}_{row['time_end']}.mp4"
#     label_num = label_to_id.get(row['label'])
    
#     output_rows.append([video_path, label_num])

# output_df = pd.DataFrame(output_rows, columns=['video_path', 'label'])
# output_df.to_csv("data/video_annotation/mimetics50/test.csv", index=False)

import os
import pandas as pd

csv_path = 'data/video_annotation/mimetics50/test.csv'
dataset_folder = '/local_datasets/mimetics'

# CSV 파일을 불러옵니다.
df = pd.read_csv(csv_path, header=None, names=['path', 'label'])

# 존재하는 파일 경로를 찾기 위해 파일 경로 리스트를 생성합니다.
existing_files = set(os.listdir(dataset_folder))

# 각 파일 경로에서 파일명만 추출하여 해당 파일이 폴더에 있는지 확인합니다.
df['exists'] = df['path'].apply(lambda x: os.path.basename(x) in existing_files)

# 존재하지 않는 파일 경로가 포함된 행을 삭제합니다.
df = df[df['exists']]

# 불필요한 'exists' 컬럼을 제거하고 결과를 저장합니다.
df = df.drop(columns=['exists'])
df.to_csv(csv_path, index=False, header=False)