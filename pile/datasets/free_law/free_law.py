import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, stream_jsonl_zst
from pathlib import Path

logger = logging.getLogger(__name__)


class FreeLaw(Dataset):
    name = "FreeLaw Project"
    
    license = "BSD 2-Clause License"

    urls = ["http://eaidata.bmk.sh/data/FreeLaw_Opinions.jsonl.zst"]

    checksum = "8a38c34f181aa121c3a7360ad63e3e8c0b1ea0913de08a4bf1b68b3eabae3e66"

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
        return list(stream_jsonl(Path(__file__).parent / "freelaw_examples.jsonl"))

    def size_on_disk(self):
        return 20975955178

    def size(self):
        return 69470526055

    def num_docs(self):
        return 4940710
