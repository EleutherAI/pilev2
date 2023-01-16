import datasets
import logging
import pathlib



logger = logging.getLogger(__name__)

def make_dummy_mixed_data():
    dummy_data = {"text":["Dummy Text"]*100+["Dummy Text 2"]*100,'meta':["{source:'dummy_data_1'}"]*100+["{source:'dummy_data_2'}"]*100}
    dummy_dataset = datasets.Dataset.from_dict(dummy_data)
    return dummy_dataset


meta_match = {
    "dummy_data_1": lambda x: "dummy_data_1" in x["meta"],
    "dummy_data_2": lambda x: "dummy_data_2" in x["meta"]
}

if __name__ == "__main__":
    dataset = make_dummy_mixed_data()
    stats_dict = {}
    output_master_dir = "./dummy_split"
    for subset_key in meta_match.keys():
        logger.info(f"Starting to filter {subset_key}")
        print(f"Starting to filter {subset_key}")
        subset_dataset = dataset.filter(meta_match[subset_key])
        length = len(subset_dataset)
        stats_dict[subset_key] = length
        logger.info(f"Length of {subset_key} is {length}")
        print(f"Length of {subset_key} is {length}")
        output_dir = pathlib.Path(output_master_dir)/subset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        subset_dataset.save_to_disk(output_dir)
        subset_dataset = datasets.load_from_disk(output_dir)
        print(subset_dataset[0])
        logger.info(f"Saved {subset_key} to {output_dir}")
        print(f"Saved {subset_key} to {output_dir}")
