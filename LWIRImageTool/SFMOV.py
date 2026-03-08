from .ImageData import ImageData
import numpy as np
from pydantic import Field, field_validator


def _get_meta_data(filename: str) -> dict:
    """
    Parse metadata from the ASCII header of an SFMOV file.

    Parameters
    ----------
    filename : str
        Path to the ``.sfmov`` file.

    Returns
    -------
    dict
        Metadata dictionary with keys such as ``'xPixls'``, ``'yPixls'``,
        ``'NumDPs'``, and ``'DaType'``.  Integer-valued fields are cast
        to ``int``.
    """
    with open(filename, "rt", errors="ignore") as f:
        meta = {}
        for line in f:
            if line[:11] == "saf_padding":
                break
            a = line[:-1].split(" ")
            if len(a) >= 2:
                meta[a[0]] = a[1]

    for key in ("xPixls", "yPixls", "NumDPs"):
        if key in meta:
            meta[key] = int(meta[key])

    return meta


def _get_data(filename: str) -> np.ndarray:
    """
    Load the raw frame data from an SFMOV file.

    Seeks past the ASCII header to the ``DATA`` marker and reads all
    subsequent binary data into a 3-D array of shape
    ``(xPixls, yPixls, frames)``.

    Parameters
    ----------
    filename : str
        Path to the ``.sfmov`` file.

    Returns
    -------
    np.ndarray
        3-D array of shape ``(cols, rows, frames)``.  dtype is
        ``np.float32`` when ``DaType == 'Flt32'``, otherwise
        ``np.uint16``.
    """
    meta = _get_meta_data(filename)

    with open(filename, "rb") as f:
        content = f.read()
        offset = content.find(b"DATA") + 6
        f.seek(offset)

        dtype = np.float32 if meta.get("DaType") == "Flt32" else np.uint16
        data = np.fromfile(f, dtype=dtype).reshape(
            -1, meta["yPixls"], meta["xPixls"]
        )

    # Transpose from (frames, y, x) -> (y, x, frames) to match workflow convention, keep memory contiguous for QT
    return np.ascontiguousarray(data.transpose(1, 2, 0))


class SFMOV(ImageData):
    """
    ``ImageData`` reader for SFMOV image sequences exported from FLIR ResearchIR.

    Reads raw digital counts (or float32 values) and sensor metadata from
    a ``.sfmov`` file using pure NumPy I/O.

    The first frame of the sequence is stored in ``raw_counts``.  The full
    multi-frame array is available via ``all_frames``.

    Parameters
    ----------
    filename : str
        Path to the ``.sfmov`` file.

    Attributes
    ----------
    raw_counts : np.ndarray
        2-D array of the **first** frame, shape ``(rows, cols)``.
    all_frames : np.ndarray
        3-D array of all frames, shape ``(frames, rows, cols)``.
    metadata : dict
        Populated metadata keys: ``'sensorType'``, ``'bitDepth'``,
        ``'horizontalRes'``, ``'verticalRes'``, ``'bands'``, ``'numFrames'``.

    Raises
    ------
    ValueError
        If *filename* is empty, not a string, or does not end with
        ``.sfmov``.
    RuntimeError
        If the file cannot be opened or parsed.

    Examples
    --------
    >>> img = SFMOV("/data/scene_001.sfmov")
    >>> img.raw_counts.shape
    (512, 640)
    >>> img.all_frames.shape
    (640, 512, 100)
    >>> img.metadata["numFrames"]
    100
    """

    filename: str = Field(
        ...,
        description="Path to the .sfmov image sequence file",
        exclude=True,
    )
    all_frames: np.ndarray = Field(
        default=None,
        description="All frames loaded from the sfmov file, shape (frames, rows, cols)",
        exclude=True,
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Filename must be a non-empty string")
        if not v.endswith(".sfmov"):
            raise ValueError("Filename must end with .sfmov")
        return v

    def __init__(self, filename: str):
        super().__init__(filename=filename)
        self.filename = filename
        self._read_sfmov(filename)

    def _read_sfmov(self, filename: str) -> None:
        """
        Read frames and metadata from an SFMOV file.

        Parameters
        ----------
        filename : str
            Path to the ``.sfmov`` file.
        """
        try:
            meta = _get_meta_data(filename)
            data = _get_data(filename)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read SFMOV file: {filename}"
            ) from e

        self.all_frames = data
        # Expose the first frame as raw_counts to match the ImageData contract
        # data shape is (x, y, frames), so index the last axis for frame 0
        self.raw_counts = data[:, :, 0]

        da_type = meta.get("DaType", "uint16")
        bit_depth = 32 if da_type == "Flt32" else 16

        self.metadata.update({
            "sensorType": "SFMOV",
            "bitDepth": bit_depth,
            "horizontalRes": int(meta.get("xPixls", data.shape[2])),
            "verticalRes": int(meta.get("yPixls", data.shape[1])),
            "bands": 1,
            "numFrames": int(data.shape[2]),
        })