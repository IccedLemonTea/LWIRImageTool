from .ImageDataConfig import ImageDataConfig
from .ENVI import ENVI
from .RJPEG import RJPEG
from .SFMOV import SFMOV
import struct


class ImageDataFactory:
    """
    Factory for creating source-agnostic ``ImageData`` objects.
    Dispatches to the correct reader (``RJPEG``, ``ENVI``, ``PySFMov``)
    based on the ``fileformat`` field of the supplied config.  Callers
    always receive an ``ImageData`` instance regardless of the underlying
    format.

    Methods
    -------
    create_from_file(config)
        Load and return an ``ImageData`` object.
    is_valid_image_file(filename, fileformat)
        Return ``True`` if *filename* is a valid file for *fileformat*.
    get_image_filetype(filename)
        Return the format identifier for *filename*, or ``None`` if unknown.

    Examples
    --------
    >>> config = ImageDataConfig(filename="/data/frame_001_R.jpg")
    >>> img = ImageDataFactory.create_from_file(config)
    >>> img.raw_counts.shape
    (512, 640)
    >>> config = ImageDataConfig(filename="/data/scene.sfmov", fileformat="sfmov")
    >>> img = ImageDataFactory.create_from_file(config)
    >>> img.all_frames.shape
    (100, 512, 640)
    """

    _MAGIC_SIGNATURES = {
        "rjpeg": (b"\xff\xd8\xff\xe0", 0),  # JPEG SOI + APP0 (JFIF) marker
        "envi":  (b"ENVI",             0),  # ENVI ASCII header
        "sfmov": (b"HdSize",           0),  # sfmov text header
    }

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
            return ENVI(config.filename, bitdepth=config.bitdepth)
        if fileformat == "rjpeg":
            return RJPEG(config.filename)
        if fileformat == "sfmov":
            return SFMOV(config.filename)
        raise ValueError(f"Unsupported file format: {fileformat}")

    @staticmethod
    def is_valid_image_file(filename: str, fileformat: str) -> bool:
        """
        Check whether a file matches the expected magic bytes for a given
        file format, rather than relying on file extension alone.

        Parameters
        ----------
        filename : str
            Path to the file to check.
        fileformat : str
            Target format identifier (``'rjpeg'``, ``'envi'``, or
            ``'sfmov'``).

        Returns
        -------
        bool
            ``True`` if the file's magic bytes match the expected signature
            for *fileformat*, ``False`` otherwise.
        """
        if fileformat not in ImageDataFactory._MAGIC_SIGNATURES:
            return False
        magic, offset = ImageDataFactory._MAGIC_SIGNATURES[fileformat]
        try:
            with open(filename, "rb") as f:
                f.seek(offset)
                header = f.read(len(magic))
            return header == magic
        except (OSError, IOError):
            return False

    @staticmethod
    def get_image_filetype(filename: str) -> str | None:
        """
        Detect the file format of an image by inspecting its magic bytes.

        Parameters
        ----------
        filename : str
            Path to the file to inspect.

        Returns
        -------
        str or None
            Format identifier (``'rjpeg'``, ``'envi'``, or ``'sfmov'``)
            if the file matches a known signature, ``None`` otherwise.
        """
        try:
            with open(filename, "rb") as f:
                for filetype, (magic, offset) in ImageDataFactory._MAGIC_SIGNATURES.items():
                    f.seek(offset)
                    if f.read(len(magic)) == magic:
                        return str(filetype)
        except (OSError, IOError):
            return None
        return None