from .ImageData import ImageData
import numpy as np
import spectral.io.envi as envi
from pydantic import Field, field_validator


class ENVI(ImageData):
    """
    ENVI image reader for LWIR thermal data.

    Loads raw counts and populates metadata from ENVI headers.
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
        Reads thermal imagery from an ENVI file.
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
