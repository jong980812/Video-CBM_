import os
import json
import pandas as pd
import pickle

base_path = "/data/datasets/kth/kth/split"
output_json = "kth_val.json"
csv_path = 'data/video_annotation/kth/val.csv'

result = []

################# train.csv 파일 읽어서 json 만들기 ###############
df = pd.read_csv(csv_path, header=None)
for file in df[0].values:
    file_id = os.path.splitext(file)[0]
    if 'walking' in file_id:
        result.append({
            "id": file_id,
            "img_path": f"{file_id}.mp4",
            "attribute_label": [0,0,0,0,0,1,1,0,0]
        })
    elif 'running' in file_id:
        result.append({
            "id": file_id,
            "img_path": f"{file_id}.mp4",
            "attribute_label": [0,1,0,0,1,1,1,0,0]
        })
    elif 'boxing' in file_id:
        result.append({
            "id": file_id,
            "img_path": f"{file_id}.mp4",
            "attribute_label": [0,1,0,1,0,0,1,0,0]
        })
    elif 'handclapping' in file_id:
        result.append({
            "id": file_id,
            "img_path": f"{file_id}.mp4",
            "attribute_label": [0,1,0,0,0,0,0,1,1]
        })
    elif 'handwaving' in file_id:
        result.append({
            "id": file_id,
            "img_path": f"{file_id}.mp4",
            "attribute_label": [1,1,1,1,0,0,0,0,1]
        })
    else:
        print(file_id, "is don't have label.")
############################################################

############# 폴더 안에 있는 비디오 이름 쭉 읽어서 만들기 ############
# for root, _, files in os.walk(base_path):
#     for file in files:
#         if file.endswith(".mp4"):
#             file_id = os.path.splitext(file)[0]
#             if 'walking' in file_id:
#                 result.append({
#                     "id": file_id,
#                     "img_path": f"{file_id}.mp4",
#                     "attribute_label": [0,0,0,0,0,1,1,0,0]
#                 })
#             elif 'running' in file_id:
#                 result.append({
#                     "id": file_id,
#                     "img_path": f"{file_id}.mp4",
#                     "attribute_label": [0,1,0,0,1,1,1,0,0]
#                 })
#             elif 'boxing' in file_id:
#                 result.append({
#                     "id": file_id,
#                     "img_path": f"{file_id}.mp4",
#                     "attribute_label": [0,1,0,1,0,0,1,0,0]
#                 })
#             elif 'handclapping' in file_id:
#                 result.append({
#                     "id": file_id,
#                     "img_path": f"{file_id}.mp4",
#                     "attribute_label": [0,1,0,0,0,0,0,1,1]
#                 })
#             elif 'handwaving' in file_id:
#                 result.append({
#                     "id": file_id,
#                     "img_path": f"{file_id}.mp4",
#                     "attribute_label": [1,1,1,1,0,0,0,0,1]
#                 })
#             else:
#                 print(file_id, "is don't have label.")
################################################################################


with open(output_json, "w") as f:
    json.dump(result, f, indent=4)

print(f"JSON 파일이 '{output_json}'로 저장되었습니다.")

pkl_file_path = f'{os.path.splitext(output_json)[0]}.pkl'
# JSON 파일을 읽어서 데이터 로드
with open(output_json, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 데이터를 Pickle 파일로 저장
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(data, pkl_file)