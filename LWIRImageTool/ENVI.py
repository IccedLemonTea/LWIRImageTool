from .ImageData import ImageData
import numpy as np
import spectral.io.envi as envi
from pydantic import Field, field_validator


class ENVI(ImageData):
    """
    ``ImageData`` reader for ENVI-format LWIR imagery.

    Reads raw digital counts and sensor metadata from an ENVI header/data
    file pair using the ``spectral`` library.

    Parameters
    ----------
    filename : str
        Path to the ENVI data file *without* the ``.hdr`` extension.
        The corresponding header is expected at ``filename + '.hdr'``.

    Attributes
    ----------
    raw_counts : np.ndarray
        2-D array of raw digital counts, shape ``(rows, cols)``.
    metadata : dict
        Populated metadata keys: ``'sensorType'``, ``'bitDepth'``,
        ``'horizontalRes'``, ``'verticalRes'``, ``'bands'``.

    Raises
    ------
    ValueError
        If *filename* is empty or not a string.

    Examples
    --------
    >>> img = ENVI("/data/scene_001")
    >>> img.raw_counts.shape
    (512, 640)
    >>> img.metadata["bitDepth"]
    16
    """


    filename: str = Field(
        ...,
        description="Path to ENVI image file without the .hdr extension",
        exclude=True
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str):
        if not v or not isinstance(v, str):
            raise ValueError("Filename must be a non-empty string")
        return v

    def __init__(self, filename: str):
        super().__init__(filename = filename)
        self.filename = filename
        self._read_envi(filename)

    def _read_envi(self, filename: str):
        """
        Read raw counts and metadata from an ENVI file pair.

        Parameters
        ----------
        filename : str
            Path to the ENVI data file (without ``.hdr`` extension).
        """

        image = envi.open(filename + ".hdr", filename)
        data = image.load()

        self.raw_counts = np.asarray(data)

        self.metadata.update({
            "sensorType": "ENVI",
            "bands": int(image.metadata.get("bands", 1)),
            "bitDepth": self.envi_dtype_to_bitdepth(
                image.metadata.get("data type")
            ),
            "horizontalRes": int(image.metadata.get("samples")),
            "verticalRes": int(image.metadata.get("lines")),
        })

    @staticmethod
    def envi_dtype_to_bitdepth(dtype_code: str | None) -> int | None:
        """
        Map an ENVI data-type code to a bit depth integer.

        Parameters
        ----------
        dtype_code : str or None
            ENVI ``data type`` field value (e.g. ``'2'`` for 16-bit integer).

        Returns
        -------
        int or None
            Bit depth corresponding to *dtype_code*, or ``None`` if the
            code is unrecognised or not provided.
        """
        mapping = {
            "1": 8,
            "2": 16,
            "3": 32,
            "4": 32,
            "5": 64,
            "6": 64,
            "9": 128,
            "12": 16,
            "13": 32,
            "14": 64,
            "15": 64,
        }
        return mapping.get(str(dtype_code)) if dtype_code else None
