import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, stream_jsonl_zst
from pathlib import Path

logger = logging.getLogger(__name__)


class EnronEmails(Dataset):
    name = "EuroParl"
    
    license = "Except where otherwise indicated, reproduction is authorised, provided that the source is acknowledged"

    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/EuroParliamentProceedings_1996_2011.jsonl.zst"]

    checksum = "6111400e7b7f75ce91fed1b5fc0a3630b8263217bd01ce75f7d8701f26ac0e98"

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
        return list(stream_jsonl(Path(__file__).parent / "euro_parl_examples.jsonl"))

    def size_on_disk(self):
        return 1475587201

    def size(self):
        return 4923130035

    def num_docs(self):
        return 69814
