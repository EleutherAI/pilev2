import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, stream_jsonl_zst
from pathlib import Path

logger = logging.getLogger(__name__)


class NIHRePORTER(Dataset):
    name = "National Institute of Health RePORTER"
    
    license = "Public domain"

    urls = ["https://mystic.the-eye.eu/public/AI/pile_v2/data/nih_reporter.jsonl.zst ", "http://eaidata.bmk.sh/data/pile_v2/NIH_ExPORTER_awarded_grant_text.jsonl.zst", "https://drive.google.com/file/d/1Sz9mFTPFa4ePYHy0AOSajUtKgVyk9mTg/view?usp=sharing"]

    checksum = "0db76318737fda6c2a2484b809bb53e9e42952c284c0bf2b8862e8428e154833"

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
        return list(stream_jsonl(Path(__file__).parent / "enron_examples.jsonl"))

    def size_on_disk(self):
        return 664022839

    def size(self):
        return 2198670684

    def num_docs(self):
        return 985651
