from .ImageDataConfig import ImageDataConfig
from .ENVI import ENVI
from .RJPEG import RJPEG


class ImageDataFactory:
    """
    Factory for creating source-agnostic ``ImageData`` objects.

    Dispatches to the correct reader (``RJPEG``, ``ENVI``, …) based on
    the ``fileformat`` field of the supplied config.  Callers always
    receive an ``ImageData`` instance regardless of the underlying format.

    Methods
    -------
    create_from_file(config)
        Load and return an ``ImageData`` object.
    is_valid_image_file(filename, fileformat)
        Return ``True`` if *filename* is a valid file for *fileformat*.

    Examples
    --------
    >>> config = ImageDataConfig(filename="/data/frame_001_R.jpg")
    >>> img = ImageDataFactory.create_from_file(config)
    >>> img.raw_counts.shape
    (512, 640)
    """

    @staticmethod
    def create_from_file(config: ImageDataConfig):
        """
        Load an image from disk and return an ``ImageData`` object.

        Parameters
        ----------
        config : ImageDataConfig
            Validated image configuration.

        Returns
        -------
        ImageData
            Populated image container (subclass depends on format).

        Raises
        ------
        ValueError
            If the file does not pass ``is_valid_image_file()`` validation,
            or if the format is not supported.
        """
        fileformat = config.fileformat.lower()

        if not ImageDataFactory.is_valid_image_file(
            config.filename, fileformat
        ):
            raise ValueError(
                f"Invalid {fileformat} file: {config.filename}"
            )

        if fileformat == "envi":
            return ENVI(
                config.filename,
                bitdepth=config.bitdepth
            )

        if fileformat == "rjpeg":
            return RJPEG(config.filename)

        raise ValueError(f"Unsupported file format: {fileformat}")

    @staticmethod
    def is_valid_image_file(filename: str, fileformat: str) -> bool:
        """
        Check whether a filename is valid for a given file format.

        Parameters
        ----------
        filename : str
            Path to the file to check.
        fileformat : str
            Target format identifier (``'rjpeg'`` or ``'envi'``).

        Returns
        -------
        bool
            ``True`` if the filename matches the expected extension for
            *fileformat*, ``False`` otherwise.
        """
        if fileformat == "rjpeg":
            return filename.endswith("_R.jpg")
        if fileformat == "envi":
            return filename.endswith(".hdr")
        return False
