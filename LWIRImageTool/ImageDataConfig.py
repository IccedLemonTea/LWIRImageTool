
from pydantic import BaseModel, Field, field_validator
from typing import Literal

ImageFormat = Literal["envi", "rjpeg"]


class ImageDataConfig(BaseModel):
    """
    Configuration for loading a single LWIR image file.

    Pass an instance of this class to ``ImageDataFactory.create_from_file()``.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the image file.  Must be a non-empty
        string; validated on construction.
    fileformat : {'rjpeg', 'envi'}, optional
        Image file format.  Default is ``'rjpeg'``.
    bitdepth : int, optional
        Bit depth of the image data.  Must be >= 1.  Default is ``16``.
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
