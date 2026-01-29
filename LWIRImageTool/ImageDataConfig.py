### ImageDataConfig Class ###
# Author : Cooper White (cjw9009@g.rit.edu)
# Date : 01/28/2026
# File : ImageDataConfig.py

from pydantic import BaseModel, Field, field_validator
from typing import Literal

ImageFormat = Literal["envi", "rjpeg"]


class ImageDataConfig(BaseModel):
    """
    Object for setting all necessary values in an image

    Variables
    ----------
    filename : str
        Path to the file containing the image the user would like to load.
    fileformat : str
        Type/format of the image files (e.g., 'rjpeg', 'envi').
    bitdepth : int
        bitdepth of the image
    """
    filename: str = Field(..., description="Path to the image file")
    fileformat: ImageFormat = Field(
        default="rjpeg",
        description="Image file format"
    )
    bitdepth: int = Field(
        default=16,
        ge=1,
        description="Bit depth for image"
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        if not isinstance(v, str) or len(v) == 0:
            raise ValueError("Filename must be a non-empty string")
        return v
