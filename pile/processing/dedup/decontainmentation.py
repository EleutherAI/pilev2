from datasets import load_dataset
from hf_clean_benchmarks.core import BenchmarkCleaner

# Benchmarks to clean
benchmarks = [
    {
        "name": "codeparrot/apps",
        "subset": "all",
        "splits": ["test"],
        "columns": ["question", "solutions"],
    },
    {
        "name": "ai2_arc",
        "subset": "ARC-Challenge",
        "splits": ["test"],
        "columns": ["question", "choices", "answerKey"],
    },
    {
        "name": "ai2_arc",
        "subset": "ARC-Easy",
        "splits": ["test"],
        "columns": ["question", "choices", "answerKey"],
    },
    {
        "name": "boolq",
        "splits": ["validation"],
        "columns": ["question", "passage"],
    },
    {
        "name": "cnn_dailymail",
        "subset": "1.0.0",
        "splits": ["test"],
        "columns": ["article", "highlights"],
    },
    {
        "name": "cnn_dailymail",
        "subset": "2.0.0",
        "splits": ["test"],
        "columns": ["article", "highlights"],
    },
    {
        "name": "cnn_dailymail",
        "subset": "3.0.0",
        "splits": ["test"],
        "columns": ["article", "highlights"],
    },
    {
        "name": "pkavumba/balanced-copa",
        "splits": ["test"],
        "columns": ["premise", "choice1", "choice2"],
    },
    {
        "name": "head_qa",
        "subset": "en",
        "splits": ["test"],
        "columns": ["qtext", "answers"],
    },
    {
        "name": "hellaswag",
        "splits": ["test"],
        "columns": ["ctx", "ctx_a", "ctx_b", "endings"],
    },
    {
        "name": "openai_humaneval",
        "splits": ["test"],
        "columns": ["prompt", "canonical_solution", "test"],
    },
    {
        "name": "lambada",
        "splits": ["test"],
        "columns": ["text"],
    },
    {
        "name": "math_qa",
        "splits": ["test"],
        "columns": ["Problem", "Rationale", "annotated_formula", "linear_formula"],
    },
    {
        "name": "mbpp",
        "subset": "full",
        "splits": ["test"],
        "columns": ["text", "code"],
    },
    {
        "name": "multi_news",
        "splits": ["test"],
        "columns": ["document", "summary"],
    },
    {
        "name": "eraser_multi_rc",
        "splits": ["test"],
        "columns": ["passage", "query_and_answer"],
    },
    {
        "name": "openbookqa",
        "subset": "main",
        "splits": ["test"],
        "columns": ["question_stem", "choices", "fact1"],
    },
    {
        "name": "openbookqa",
        "subset": "additional",
        "splits": ["test"],
        "columns": ["question_stem", "choices", "fact1"],
    },
    {
        "name": "piqa",
        "splits": ["test"],
        "columns": ["goal", "sol1", "sol2"],
    },
    {
        "name": "pubmed_qa",
        "subset": "pqa_labeled",
        "splits": ["test"],
        "columns": ["question", "context", "long_answer"],
    },
    {
        "name": "pubmed_qa",
        "subset": "pqa_unlabeled",
        "splits": ["test"],
        "columns": ["question", "context", "long_answer"],
    },
    {
        "name": "pubmed_qa",
        "subset": "pqa_artificial",
        "splits": ["test"],
        "columns": ["question", "context", "long_answer"],
    },
    {
        "name": "glue",
        "subset": "rte",
        "splits": ["test"],
        "columns": ["sentence1", "sentence2"],
    },
    {
        "name": "sciq",
        "splits": ["test"],
        "columns": ["question", "support"],
    },
    {
        "name": "squad",
        "splits": ["validation"],
        "columns": ["context", "question", "answers"],
    },
    {
        "name": "squad_v2",
        "splits": ["validation"],
        "columns": ["context", "question", "answers"],
    },
    {
        "name": "super_glue",
        "subset": "cb",
        "splits": ["test"],
        "columns": ["premise", "hypothesis"],
    },
    {
        "name": "super_glue",
        "subset": "wic",
        "splits": ["test"],
        "columns": ["sentence1", "sentence2"],
    },
    {
        "name": "super_glue",
        "subset": "wsc",
        "splits": ["test"],
        "columns": ["text"],
    },
    {
        "name": "trivia_qa",
        "subset": "rc",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "rc.nocontext",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "rc.web",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "rc.web.nocontext",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "rc.wikipedia",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "rc.wikipedia.nocontext",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "unfiltered",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "trivia_qa",
        "subset": "unfiltered.nocontext",
        "splits": ["test"],
        "columns": ["question", "entity_pages", "search_results"],
    },
    {
        "name": "wikitext",
        "subset": "wikitext-103-raw-v1",
        "splits": ["test"],
        "columns": ["text"],
    },
    {
        "name": "winogrande",
        "subset": "winogrande_debiased",
        "splits": ["test"],
        "columns": ["sentence"],
    },
    {
        "name": "wino_bias",
        "subset": "type1_pro",
        "splits": ["test"],
        "columns": ["tokens"],
    },
    {
        "name": "wino_bias",
        "subset": "type1_anti",
        "splits": ["test"],
        "columns": ["tokens"],
    },
    {
        "name": "wino_bias",
        "subset": "type2_pro",
        "splits": ["test"],
        "columns": ["tokens"],
    },
    {
        "name": "wino_bias",
        "subset": "type2_anti",
        "splits": ["test"],
        "columns": ["tokens"],
    },
    {
        "name": "yelp_review_full",
        "splits": ["test"],
        "columns": ["text"],
    },
]

# MATH=???
# MMMLU=???
# SuperGLUE=???
# Benchmarks to clean
# benchmar`ks = [
#     {
#         "name": "openai_humaneval",
#         "splits": ["test"],
#         "columns": ["prompt", "canonical_solution", "test"],
#     },
#     {
#         "name": "lambada",
#         "splits": ["test"],
#         "columns": ["text"],
#     },
# ]`

cleaner = BenchmarkCleaner(benchmarks, threshold=0.1, num_perm=128)

# load your dataset
dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")

# clean the dataset
cleaned_dataset = cleaner.clean(dataset, column="content")