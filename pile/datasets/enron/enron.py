import logging
from ...templates import Dataset
from ...file_utils import stream_jsonl, stream_jsonl_zst
from pathlib import Path

logger = logging.getLogger(__name__)


class EnronEmails(Dataset):
    name = "Enron Emails"
    
    license = "Public domain"

    urls = ["http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"]

    checksum = "6968dd2d6d9c4328ee3b77b263aad38401b77c326f693ce051c98a3f215bf583"

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
        return 233861493

    def size(self):
        return 945212874

    def num_docs(self):
        return 517401
