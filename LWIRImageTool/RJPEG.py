from .ImageData import ImageData
import numpy as np
import subprocess
import PIL.Image
import io
from pydantic import Field

class RJPEG(ImageData):
    """
    ``ImageData`` reader for FLIR radiometric JPEG (RJPEG) imagery.

    Extracts the embedded 16-bit raw thermal image from a FLIR ``_R.jpg``
    file using ``exiftool``, then decodes it with Pillow.

    Parameters
    ----------
    filename : str
        Path to the RJPEG file.  Must end with ``'_R.jpg'``.

    Attributes
    ----------
    raw_counts : np.ndarray
        2-D ``uint16`` array of raw digital counts, shape ``(rows, cols)``.
    metadata : dict
        Populated metadata keys: ``'sensorType'`` (``'RJPEG'``),
        ``'bitDepth'`` (``16``), ``'horizontalRes'``, ``'verticalRes'``,
        ``'bands'`` (``1``).

    Raises
    ------
    ValueError
        If *filename* does not end with ``'_R.jpg'``.
    RuntimeError
        If ``exiftool`` fails or the embedded thermal blob cannot be decoded.

    Notes
    -----
    Requires ``exiftool`` to be installed and available on ``PATH``.

    Examples
    --------
    >>> img = RJPEG("/data/20260210_122136_126_LWIR_R.jpg")
    >>> img.raw_counts.shape
    (512, 640)
    >>> img.raw_counts.dtype
    dtype('uint16')
    """
    filename: str = Field(..., exclude=True)

    def __init__(self, filename: str):
        # Pass filename to BaseModel constructor
        super().__init__(filename=filename)
        self._read_rjpeg(filename)

    def _read_rjpeg(self, filename: str):
        """
        Extract and decode the raw thermal image from an RJPEG file.

        Calls ``exiftool -b -RawThermalImage`` as a subprocess to extract
        the embedded 16-bit image blob, then decodes it with Pillow.

        Parameters
        ----------
        filename : str
            Path to the ``_R.jpg`` file.

        Raises
        ------
        ValueError
            If *filename* does not end with ``'_R.jpg'``.
        RuntimeError
            If the ``exiftool`` subprocess returns a non-zero exit code,
            or if Pillow cannot decode the resulting blob.
        """
        if not filename.endswith("_R.jpg"):
            raise ValueError(f"Not an RJPEG file: {filename}")

        try:
            cmd = ["exiftool", "-b", "-RawThermalImage", filename]
            result = subprocess.run(cmd, capture_output=True, check=True)
            blob = result.stdout

            raw = PIL.Image.open(io.BytesIO(blob))
            img = np.asarray(raw, dtype=np.uint16)

            self.raw_counts = img
            self.metadata.update({
                "sensorType": "RJPEG",
                "bitDepth": 16,
                "horizontalRes": img.shape[1],
                "verticalRes": img.shape[0],
                "bands": 1,
            })

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Exiftool failed for {filename}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to decode thermal image: {filename}") from e
