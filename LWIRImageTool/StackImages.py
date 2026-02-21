import numpy as np
from .ImageDataFactory import ImageDataFactory
from .ImageDataConfig import ImageDataConfig
import os


def stack_images(directory, filetype, progress_cb=None):
    """
    Load all valid images from a directory and stack them along a time axis.

    Files are sorted lexicographically (which preserves chronological order
    when FLIR timestamp filenames are used).  The output array dtype matches
    the dtype of the first image loaded.

    Parameters
    ----------
    directory : str
        Path to the directory containing the image sequence.
    filetype : str
        Image format identifier passed to ``ImageDataFactory``.
        Supported values: ``'rjpeg'``, ``'envi'``.
    progress_cb : callable or None, optional
        Progress callback invoked as
        ``progress_cb(phase='loading', current=int, total=int)``
        after each frame is loaded.  ``None`` disables callbacks.
        Default is ``None``.

    Returns
    -------
    image_stack : np.ndarray
        3-D array of stacked frames, shape ``(rows, cols, num_frames)``.
        Dtype matches the raw counts dtype of the source images.

    Raises
    ------
    IndexError
        If no valid image files are found in *directory*.
    ValueError
        If any image file fails validation inside ``ImageDataFactory``.

    Examples
    --------
    >>> stack = stack_images("/data/cal_run", filetype="rjpeg")
    >>> stack.shape
    (512, 640, 1200)
    >>> stack.dtype
    dtype('uint16')
    """
    
    ### PREALLOCATING SPACE FOR VECTORS AND SORTING DIR FOR IMAGES ###
    factory = ImageDataFactory()
    file_list = sorted(os.listdir(directory))
    image_list = []

    for f in file_list:
        if factory.is_valid_image_file(f, filetype):
            image_list.append(f)

    config = ImageDataConfig(filename = os.path.join(directory, image_list[0]), fileformat= filetype)
    first_src = factory.create_from_file(config)

    rows, cols = first_src.raw_counts.shape
    num_frames = len(image_list)

    image_stack = np.zeros((rows, cols, num_frames),
                           dtype=first_src.raw_counts.dtype)
    image_stack[:, :, 0] = first_src.raw_counts

    # Optional callback update for GUI
    idx = 0
    if progress_cb:
        progress_cb(phase="loading", current=idx + 1, total=num_frames)
        idx += 1

    # Stacking all images in the target directory
    for idx, file in enumerate(image_list):
        file_path = os.path.join(directory, file)
        config.filename = file_path
        src = factory.create_from_file(config)
        image_stack[:, :, idx] = np.array(src.raw_counts)

        # Optional callback for GUI
        if progress_cb:
            progress_cb(phase="loading", current=idx + 1, total=num_frames)
            idx += 1

    return image_stack


if __name__ == "__main__":
    from .StackImages import stack_images
    import numpy as np

    dir = "/home/cjw9009/Desktop/Senior_Project/FLIRSIRAS_CalData/20251202_1700/"

    print(f"Stacking images...")
    stack = stack_images(directory=dir, filetype='rjpeg')

    np.save("20251202_1700_imagestack_new.npy", stack)
