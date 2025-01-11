import csv
from collections import defaultdict
import os
import data_utils
import json

def analyze_and_replace_concepts(file_path, stp_concepts):
    # 컨셉 번호를 컨셉명으로 매핑하는 딕셔너리 생성
    concept_id_to_name = {i: stp_concepts[i] for i in range(len(stp_concepts))}
    
    class_concept_dict = defaultdict(lambda: defaultdict(int))

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            class_id = int(row[0])
            concept_ids = map(int, row[1:])

            for concept_id in concept_ids:
                concept_name = concept_id_to_name.get(concept_id, f"Unknown:{concept_id}")
                class_concept_dict[class_id][concept_name] += 1

        return class_concept_dict
    
def contribution_analysis(args):

    dataset = args.data_set
    cls_file = data_utils.LABEL_FILES[dataset]

    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
        
    with open(os.path.join(args.debug, 'spatial', "concepts.txt"), "r") as f:
        s_concepts = f.read().split("\n")
        s_concepts = ["S:{}".format(word) for word in s_concepts]

    with open(os.path.join(args.debug, 'temporal', "concepts.txt"), "r") as f:
        t_concepts = f.read().split("\n")
        t_concepts = ["T:{}".format(word) for word in t_concepts]

    with open(os.path.join(args.debug, 'place', "concepts.txt"), "r") as f:
        p_concepts = f.read().split("\n")
        p_concepts = ["P:{}".format(word) for word in p_concepts]

    stp_concepts = s_concepts + t_concepts + p_concepts

    class_concept_contributions_named = analyze_and_replace_concepts(os.path.join(args.debug,'class_concept_contribution.csv'), stp_concepts)

    with open(os.path.join(args.debug,'contribution.txt'), "w") as output_file:
        for class_id, concept_dict in class_concept_contributions_named.items():
            high_activating = dict(sorted(
                {concept: count for concept, count in concept_dict.items() if count >= 2}.items(),
                key=lambda item: item[1],
                reverse=True
            ))
            intervention_candidates = [concept for concept, count in concept_dict.items() if count == 1]

            output_file.write(f"Class {class_id}: {classes[class_id]}\n")
            output_file.write(f"high activating: {high_activating}\n")
            output_file.write(f"intervention candidates: {intervention_candidates}\n\n")