### ImageDataFactory Class ###
# Author : Cooper White
# Date : 09/30/2025
# File : ImageDataFactory.py

import LWIRImageTool
from typing import Literal

ImageFormat = Literal["envi", "rjpeg"]


class ImageDataFactory(object):
    """Factory for creating source-agnostic ImageData objects."""

    @staticmethod
    def create_from_file(
        filename: str,
        fileformat: ImageFormat = "envi",
        bitdepth: int = 16,
    ):
        fileformat = fileformat.lower()

        if not ImageDataFactory.is_valid_image_file(filename, fileformat):
            raise ValueError(f"Invalid {fileformat} file: {filename}")

        if fileformat == "envi":
            return LWIRImageTool.ENVI(filename, bitdepth=bitdepth)

        if fileformat == "rjpeg":
            return LWIRImageTool.RJPEG(filename)

        raise ValueError(f"Unsupported file format: {fileformat}")

    @staticmethod
    def is_valid_image_file(filename: str, fileformat: str) -> bool:
        if fileformat == "rjpeg":
            return filename.endswith("_R.jpg")
        elif fileformat == "envi":
            return filename.endswith(".hdr")
        else:
            return False
