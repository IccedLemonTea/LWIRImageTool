from .ImageData import ImageData
import numpy as np
import subprocess
import PIL.Image
import io
from pydantic import Field

class RJPEG(ImageData):
    filename: str = Field(..., exclude=True)

    def __init__(self, filename: str):
        # Pass filename to BaseModel constructor
        super().__init__(filename=filename)
        self._read_rjpeg(filename)

    def _read_rjpeg(self, filename: str):
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
