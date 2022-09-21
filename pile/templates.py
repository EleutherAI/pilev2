from abc import ABC, abstractmethod
from .utils import (
    directory_size,
    download,
    disk_cache,
    utf8len,
    pile_cache_dir,
    component_exists,
    schema_from_examples,
)
from tqdm import tqdm
from typing import Generator, List, Optional, Union
from pathlib import Path
from logging import getLogger
from pprint import pformat

logger = getLogger(__name__)


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the dataset.
        """
        pass
    
    @property
    @abstractmethod
    def license(self) -> str:
        """
        The license that the data is under.
        """
        pass

    @property
    @abstractmethod
    def urls(self) -> List[str]:
        """Returns a list of urls to download the dataset.

        The most trusted miror should be first.
        """
        pass

    @property
    @abstractmethod
    def checksum(self) -> str:
        """
        The sha256 checksum of the dataset.
        """
        pass

    @abstractmethod
    def replicate(self):
        """Replicates the scraping & filtering process used to gather the
        dataset.

        If the dataset is hosted by an outside source and there's no
        filtering steps, or it's not possible to replicate the scraping
        process, this method should just call self.download()
        """
        pass

    @abstractmethod
    def documents(self) -> Generator[dict, None, None]:
        """A generator producing all documents in the dataset, in dictionary
        form.

        Each document should be a dictionary with, at minimum, a 'text'
        key containing the text of the document.
        """
        pass

    @abstractmethod
    def paths(self) -> Generator[Union[str, Path], None, None]:
        """A generator producing the paths on disk of all files in the dataset.

        If the dataset is not on disk, this should still return the
        *expected* filepaths when the dataset is downloaded.
        """
        pass

    @abstractmethod
    def examples(self) -> List[dict]:
        """returns a small amount of example documents from the dataset.

        Should be callable before downloading the whole dataset.
        """
        pass

    @abstractmethod
    def size_on_disk(self) -> int:
        """Return the size of the dataset on disk in bytes.

        This should be hardcoded for each dataset, so the user can know the size of the dataset
        before downloading it.

        We can verify the hardcoded value by comparing to Dataset._size_on_disk()
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the uncompressed size of just the text portion of the dataset
        (i.e the size that will be trained on, excluding metadata), in bytes.

        This should be hardcoded for each dataset, we can verify the
        hardcoded value by comparing to Dataset._size()
        """
        pass

    def download(self, force=False):
        """Downloads the final hosted dataset. If necessary, this function
        should also extract the dataset into a format that can be streamed by
        lm_dataformat, and then remove the compressed archive.

        Args:
            force: If True, redownload the dataset even if it already exists.
        """
        out_path = self.dataset_dir() / str(Path(self.url).name)
        logger.info(f"Downloading {self.name} to {out_path}")
        return download(
            url=self.url,
            out_path=out_path,
            mirrors=self.mirrors,
            checksum=self.checksum,
            force=force,
        )

    def _size_on_disk(self) -> int:
        return directory_size(self.dataset_dir())

    def _size(self) -> int:
        return sum(
            utf8len(doc["text"])
            for doc in tqdm(self.documents(), desc=f"Calculating size of {self.name}")
        )

    @disk_cache()
    def num_docs(self) -> int:
        """Return an estimate of the number of documents in the dataset.

        Implementations may use a faster, less accurate estimate.
        """
        return len(
            list(
                map(
                    lambda x: None,
                    tqdm(self.documents(), f"Counting documents of {self.name}"),
                )
            )
        )

    @property
    def url(self) -> str:
        """
        Return the url to download the dataset.
        """
        return list(self.urls)[0] if len(list(self.urls)) > 0 else None

    @property
    def mirrors(self) -> List[str]:
        """
        Return a list of urls to download the dataset from if the initial url
        fails.
        """
        return self.urls[1:]

    @property
    def schema(self) -> dict:
        """Return the schema of the dataset.

        This is generated automatically based on self.examples()
        """
        return schema_from_examples(self.examples())

    def info(self) -> dict:
        """
        Return a dict with information about the dataset.
        """
        info_dict = {
            "name": self.name,
            "url": self.url,
            "checksum": self.checksum,
            "size": self.size(),
            "size_on_disk": self.size_on_disk(),
            "paths": list(str(p) for p in self.paths()),
            "num_docs": self.num_docs(),
            "schema": self.schema,
        }
        if self.mirrors:
            info_dict["mirrors"] = self.mirrors
        return info_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(\n" + pformat(self.info()) + "\n)"

    def dataset_dir(self) -> Path:
        """
        Return the path to the parent directory of the dataset.
        """
        return pile_cache_dir() / self.name.lower().replace(" ", "_").replace(
            "(", ""
        ).replace(")", "")

    def exists(self) -> bool:
        """
        Return whether the complete dataset exists on disk.
        """
        return all(component_exists(p) for p in self.paths())

    def raise_if_not_exists(self):
        """
        Raise an exception if the dataset does not exist on disk.
        """
        if not self.exists():
            raise Exception(
                f"{self.name} dataset not found, please call {self.__class__.__name__}.download()"
            )
