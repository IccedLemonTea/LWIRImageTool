from .ImageDataConfig import ImageDataConfig
from .ENVI import ENVI
from .RJPEG import RJPEG


class ImageDataFactory:
    """Factory for creating source-agnostic ImageData objects."""

    @staticmethod
    def create_from_file(config: ImageDataConfig):
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
        if fileformat == "rjpeg":
            return filename.endswith("_R.jpg")
        if fileformat == "envi":
            return filename.endswith(".hdr")
        return False
