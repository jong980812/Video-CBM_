# # txt 파일 읽기
# with open("/data/jong980812/project/Video-CBM-two-stream/SUN_places.txt", "r") as file:
#     lines = file.readlines()



# # 전처리 및 변환
# processed_lines = []
# for line in lines:
#     line = line.strip().strip("'")  # 개행 문자와 마지막 ' 제거
#     parts = line.split("/")  # '/'를 기준으로 분리
#     main_context = parts[2]
    
#     # 추가 설명이 있는 경우
#     if len(parts) > 3:
#         sub_context = parts[3].replace('_', ' ')
#         # 'indoor'와 'outdoor' 처리
#         if sub_context == "indoor":
#             processed_line = f"inside the {main_context.replace('_', ' ')}"
#         elif sub_context == "outdoor":
#             processed_line = f"outside the {main_context.replace('_', ' ')}"
#         else:
#             # 일반 형용사 처리
#             processed_line = f"{sub_context} {main_context.replace('_', ' ')}"
#     else:
#         # 설명 없는 경우 기본 장소 명칭만
#         processed_line = main_context.replace('_', ' ')

#     processed_lines.append(processed_line)

# # 결과 파일로 저장
# with open("processed_locations.txt", "w") as file:
#     for line in processed_lines:
#         file.write(f"{line}\n")

# 파일 경로 지정
file_path = "/data/jong980812/project/Video-CBM-two-stream/data/concept_sets/llava_ver2/k400_obejct.txt"  # 파일 경로를 실제 파일 경로로 변경하세요.

# 파일 열어서 중복 제거 후 저장
with open(file_path, "r") as f:
    lines = f.read().splitlines()  # 각 줄을 리스트로 읽기

# 중복 제거
unique_lines = list(set(lines))

# 정렬해서 저장하고 싶다면 아래 줄을 추가하세요
# unique_lines.sort()

# 중복 제거된 내용을 다시 파일에 저장
with open(file_path, "w") as f:
    f.write("\n".join(unique_lines))

print("중복 제거 후 파일이 저장되었습니다.")