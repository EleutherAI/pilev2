import pickle
import numpy as np
path = "/fsx/shared/hf_data_pilev2_small_text/stats_dict.pkl"
with open(path, "rb") as f:
    stats_dict = pickle.load(f)

print(stats_dict.keys())
for dataset, stats in stats_dict.items():
    print(dataset)
    # print(stats.keys())
    # word_min, word_max = np.quantile(stats["word_count"]["lst"], [0.01, 0.99])
    # print(f"word_min: {word_min}, word_max: {word_max}")
    # perplexity_max = np.quantile(stats["perplexity"]["lst"], 0.99)
    # print(f"perplexity_max: {perplexity_max}")
    # flagged_ratio_max = np.quantile(stats["flag_words_ratio"]["lst"], 0.99)
    # print(f"flagged_ratio_max: {flagged_ratio_max}")