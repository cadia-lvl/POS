dependencies = ["pos"]

import tarfile
from logging import getLogger
from pathlib import Path

import torch

log = getLogger(__name__)

CLARIN_URL = "http"


def _get_model_location(model_dir_or_url: str, model_name: str, force_download: bool) -> Path:
    """Returns the Path of the model on the local machine.

    Args:
        model_dir_or_url: If startswith("http") then we will (maybe) download the model. Otherwise we will load a local directory.

    Returns:
        A Path to the model."""
    if model_dir_or_url.startswith("http"):
        cache_dir = Path(torch.hub.get_dir())
        download_location = cache_dir / f"{model_name}.tar.gz"

        need_extraction = False
        if not download_location.exists() or force_download:
            log.debug("Downloading model")
            torch.hub.download_url_to_file(model_dir_or_url, download_location)
            need_extraction = True

        model_dir = cache_dir / model_name
        if not model_dir.exists() or need_extraction:
            model_dir.mkdir(exist_ok=True)
            # Unpack the model
            tar = tarfile.open(download_location, "r:gz")
            log.debug("Extracting model")
            tar.extractall(path=model_dir)
            log.debug("Done extracting model")
            tar.close()
    else:
        model_dir = Path(model_dir_or_url)
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} does not exist")
    return model_dir


def lemma(model_dir_or_url="http://localhost:8000/lemma.tar.gz", device="cpu", force_download=False, *args, **kwargs):
    """
    Lemmatizer for Icelandic.

    model_dir_or_url (str): Default= The location of a model. Can be a URL: http://CLARIN.eu or a local folder which contains the neccessary files for loading a model.
    force_download (bool): Set to True if model should be re-downloaded.
    """
    return _load_model(model_dir_or_url, "lemma", device, force_download, *args, **kwargs)


def tag(model_dir_or_url="http://localhost:8000/pos.tar.gz", device="cpu", force_download=False, *args, **kwargs):
    """
    Part-of-Speech tagger for Icelandic.

    model_dir_or_url (str): Default= The location of a model. Can be a URL: http://CLARIN.eu or a local folder which contains the neccessary files for loading a model.
    force_download (bool): Set to True if model should be re-downloaded.
    """
    return _load_model(model_dir_or_url, "pos", device, force_download, *args, **kwargs)


def tag_large(
    model_dir_or_url="http://localhost:8000/pos-large.tar.gz", device="cpu", force_download=False, *args, **kwargs
):
    """
    A large Part-of-Speech tagger for Icelandic.

    model_dir_or_url (str): Default= The location of a model. Can be a URL: http://CLARIN.eu or a local folder which contains the neccessary files for loading a model.
    force_download (bool): Set to True if model should be re-downloaded.
    """
    return _load_model(model_dir_or_url, "pos-large", device, force_download, *args, **kwargs)


def _load_model(
    model_dir_or_url="http://localhost:8000/pos.tar.gz",
    model_name="",
    device="cpu",
    force_download=False,
    *args,
    **kwargs,
):
    """
    Part-of-Speech tagger for Icelandic.

    model_dir_or_url (str): Default= The location of a model. Can be a URL: http://CLARIN.eu or a local folder which contains the neccessary files for loading a model.
    force_download (bool): Set to True if model should be redownloaded.
    """
    from pos import Tagger

    model_location = _get_model_location(
        model_dir_or_url=model_dir_or_url, model_name=model_name, force_download=force_download
    )
    tagger = Tagger(str(model_location), device=device)

    return tagger
