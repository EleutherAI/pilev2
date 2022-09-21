from .dm_mathematics import DeepMindMathematics
from .enron import EnronEmails
from .euro_parl import EuroParl
from .freelaw import FreeLaw
from .grade_school_math import *
from .philpapers import PhilPapers
from .project_gutenberg import ProjectGutenberg
from .wikipedia import Wikipedia

DATASETS = {
    "dm_mathematics": DeepMindMathematics,
    "enron_emails": EnronEmails,
    "euro_parl": EuroParl,
    "free_law": FreeLaw,
    "grade_school_math": GradeSchoolMath,
    "grade_school_math_no_calc": GradeSchoolMathNoCalc,
    "nih_reporter": NIHRePORTER,
    "phil_papers": PhilPapers,
    "project_gutenberg": ProjectGutenberg,
    "wikipedia": Wikipedia,
}


def download_all():
    """Downloads all datasets in the DATASETS registry to the cache dir.

    the cache dir defaults to ~/pile_datasets, but can be overridden by
    setting the environment variable PILE_CACHE_DIR.
    """
    for dataset in DATASETS.values():
        dataset().download()


def replicate_all():
    """Replicates all datasets in the DATASETS registry, saving the final
    dataset to the cache dir.

    WARNING: This is a very slow operation, and you probably don't need to run this.

    the cache dir defaults to ~/pile_datasets, but can be overridden by setting the
    environment variable PILE_CACHE_DIR.
    """
    for dataset in DATASETS.values():
        dataset().replicate()


def list_datasets():
    """
    Lists all datasets in the DATASETS registry.
    """
    for name in DATASETS.keys():
        print(name)
