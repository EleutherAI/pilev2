import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, stream_jsonl_zst
from pathlib import Path

logger = logging.getLogger(__name__)


class DMMathematics(Dataset):
    name = "DeepMind Mathematics Dataset"
    
    license = "MIT License"

    urls = [""]

    checksum = "def638343403cb9ed60437d6b684c859dd23b72779f5cc5661b0a31e67c58576"

    def replicate(self):
        logger.info(f"{self.name} cannot be replicated - downloading from source")
        self.download()

    def documents(self):
        self.raise_if_not_exists()
        return stream_jsonl_zst(list(self.paths())[0])

    def paths(self):
        paths = [str(self.dataset_dir() / str(Path(self.url).name))]
        for path in paths:
            yield path

    def examples(self):
        return list(stream_jsonl(Path(__file__).parent / "dm_mathematics_examples.jsonl"))

    def size_on_disk(self):
        return -1

    def size(self):
        return 8316165951

    def num_docs(self):
        return 1014997
