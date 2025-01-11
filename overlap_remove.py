# 파일 경로 설정
k400_object_path = "data/concept_sets/llava_ver2/k400_obejct.txt"
processed_locations_path = "processed_locations.txt"
output_path = "data/concept_sets/llava_ver2/k400_object_filtered.txt"

# processed_locations.txt의 모든 행 읽기
with open(processed_locations_path, "r") as f:
    processed_locations = set(line.strip() for line in f)

# k400_object.txt에서 중복 행을 제외하고 새로운 파일에 저장
with open(k400_object_path, "r") as f, open(output_path, "w") as out_f:
    for line in f:
        if line.strip() not in processed_locations:
            out_f.write(line)

print("중복 행이 제거된 파일이 'k400_object_filtered.txt'로 저장되었습니다.")