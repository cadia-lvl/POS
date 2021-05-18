dependencies = ["pos"]

import tarfile
from logging import getLogger
from pathlib import Path

import torch

log = getLogger(__name__)

CLARIN_URL = "http"


def _get_model_location(model_dir_or_url: str, force_download: bool) -> Path:
    """Returns the Path of the model on the local machine.

    Args:
        model_dir_or_url: If startswith("http") then we will (maybe) download the model. Otherwise we will load a local directory.

    Returns:
        A Path to the model."""
    if model_dir_or_url.startswith("http"):
        cache_dir = Path(torch.hub.get_dir())
        download_location = cache_dir / "pos.tar.gz"

        need_extraction = False
        if not download_location.exists() or force_download:
            log.debug("Downloading model")
            torch.hub.download_url_to_file(model_dir_or_url, download_location)
            need_extraction = True

        model_dir = cache_dir / "pos"
        model_location = model_dir / "tagger.pt"
        if not model_location.exists() or need_extraction:
            model_dir.mkdir(exist_ok=True)
            # Unpack the model
            tar = tarfile.open(download_location, "r:gz")
            log.debug("Extracting model")
            tar.extractall(path=model_dir)
            log.debug("Done extracting model")
            tar.close()
        assert model_location.exists(), f"{model_location} should exist after extracting."
    else:
        model_location = Path(model_dir_or_url) / "tagger.pt"
        if not model_location.exists():
            raise FileNotFoundError(f"{model_location} does not exist")
    return model_location


def pos(model_dir_or_url="http://localhost:8000/pos.tar.gz", force_download=False, *args, **kwargs):
    """
    Part-of-Speech tagger for Icelandic.

    model_dir_or_url (str): Default= The location of a model. Can be a URL: http://CLARIN.eu or a local folder which contains the neccessary files for loading a model.
    force_download (bool): Set to True if model should be redownloaded.
    """
    from pos import Tagger

    model_location = _get_model_location(model_dir_or_url=model_dir_or_url, force_download=force_download)
    tagger = Tagger(model_location)
    return tagger
