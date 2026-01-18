import numpy as np
import LWIRImageTool
import os


def stack_images(directory, filetype, progress_cb=None):
    """
    Reads all images from a directory and stacks them along a third dimension.

    The third dimension corresponds to time, ordered by the file timestamps.

    Parameters
    ----------
    directory : str
        Path to the directory containing blackbody images.
    filetype : str
        Type/format of the image files.
    progress_cb : callable, optional
        Callback function for GUI progress updates.

    Returns
    -------
    image_stack : np.ndarray
        3D array containing stacked images with shape (rows, cols, num_frames).
    """
    ### PREALLOCATING SPACE FOR VECTORS AND SORTING DIR FOR IMAGES ###
    factory = LWIRImageTool.ImageDataFactory()
    file_list = sorted(os.listdir(directory))
    image_list = []

    for f in file_list:
        if factory.is_valid_image_file(f,filetype):
            image_list.append(f)

    first_src = factory.create_from_file(os.path.join(directory, image_list[0]), filetype)

    rows, cols = first_src.raw_counts.shape
    num_frames = len(image_list)

    image_stack = np.zeros((rows, cols, num_frames), dtype=first_src.raw_counts.dtype)
    image_stack[:,:,0] = first_src.raw_counts

    # Optional callback update for GUI
    idx = 0
    if progress_cb:
        progress_cb(phase="loading", current=idx + 1, total=num_frames)
        idx += 1

    # Stacking all images in the target directory
    for idx, file in enumerate(image_list):
        file_path = os.path.join(directory, file)
        src = factory.create_from_file(file_path, filetype)
        image_stack[:,:,idx] = np.array(src.raw_counts)

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
    stack = stack_images(directory = dir, filetype = 'rjpeg')



    np.save("20251202_1700_imagestack_new.npy", stack)