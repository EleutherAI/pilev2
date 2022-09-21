import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, stream_jsonl_zst
from pathlib import Path

logger = logging.getLogger(__name__)


class PhilPapers(Dataset):
    name = "PhilPapers"
    
    license = "Open Access"

    urls = ["http://eaidata.bmk.sh/data/phil_papers.jsonl.zst"]

    checksum = "9311a57fcbde8dd832e954821bdf0e1f3e2899d9567f6c3b5d7a2d1161fa3e7d"

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
        return list(stream_jsonl(Path(__file__).parent / "philpapers_examples.jsonl"))

    def size_on_disk(self):
        return 985311313

    def size(self):
        return 3270636340

    def num_docs(self):
        return 42464
