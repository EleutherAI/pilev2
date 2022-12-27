import argparse
import json
import os

# print number of cores
print(os.cpu_count())

from datasets import concatenate_datasets, load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--grouped_dup_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

with open (args.grouped_dup_path, "r") as f:
    data = json.load(f)

print(data)
in_common = data.pop("common_group")
for group_name in ["group_1", "group_2"]:
    # concatenate_datasets([data1, data2])
    group = []
    for name in data[group_name][:2]:
        ds_path = os.path.join(args.dataset_path, name)
        ds = load_from_disk(ds_path)
        ds = ds.remove_columns(
            [
                'check_char_repetition_criteria',
                'check_flagged_words_criteria',
                'check_stop_word_ratio_criteria'
            ]
        )
        group.append(ds_path)
    for name in in_common:
        ds_path = os.path.join(args.dataset_path, name)
        ds = load_from_disk(ds_path)
        ds = ds.remove_columns(
            [
                'check_char_repetition_criteria',
                'check_flagged_words_criteria',
                'check_stop_word_ratio_criteria'
            ]
        )
        group.append(ds_path)
    ds = concatenate_datasets(group)

    #     print(ds_path)

    #     group.append(ds_path)
    
    # ds = concatenate_datasets(group)

# data1 = load_from_disk("PileV2Reddit2020_ver2/PileV2Reddit2020_0")
# data1 = data1.remove_columns(['check_char_repetition_criteria', 'check_flagged_words_criteria', 'check_stop_word_ratio_criteria'])
# print(data1)
# data2 = load_from_disk("PileV2Reddit2020_ver2/PileV2Reddit2020_1")
# data2 = data2.remove_columns(['check_char_repetition_criteria', 'check_flagged_words_criteria', 'check_stop_word_ratio_criteria'])
# print(data2)
# data =  concatenate_datasets([data1, data2])
# print(data)

# dedup_pilev2 = deduplicate_dataset(data)[0]
# dedup_pilev2.save_to_disk("local_dedup/PileV2Reddit2020_ver2")

# for group in data:
#     for ds in group:
#         print(ds)