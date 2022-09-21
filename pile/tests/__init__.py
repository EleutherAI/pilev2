def test_dataset(dataset, run_download=False, run_replicate=False):
    """Tests all the properties / functions in a dataset are working without
    error.

    TODO: write proper tests
    """
    print("repr: ")
    print(dataset)

    print("\nname: ")
    print(dataset.name)

    print("\nurls & url: ")
    print(dataset.urls)
    print(dataset.url)

    print("\nchecksum: ")
    print(dataset.checksum)

    if run_download:
        print("\ndownload: ")
        dataset.download(force=True)

    if run_replicate:
        print("\nreplicate: ")
        dataset.replicate()

    print("\ndocuments: ")
    print(next(dataset.documents()))

    print("\n paths: ")
    print(next(dataset.paths()))

    print("\nexamples: ")
    print(dataset.examples())

    print("\nreturned size: ")
    print(dataset.size())

    print("\nActual size: ")
    print(dataset._size())

    print("\nreturned size on disk: ")
    print(dataset.size_on_disk())

    print("\nActual size on disk: ")
    print(dataset._size_on_disk())

    print("\nnum_docs: ")
    print(dataset.num_docs())

    print("\nschema: ")
    print(dataset.schema)

    print("\nmirrors: ")
    print(dataset.mirrors)

    print("\ndataset_dir: ")
    print(dataset.dataset_dir())

    print("\nexists: ")
    print(dataset.exists())
    try:
        dataset.raise_if_not_exists()
    except Exception as e:
        print(e)
