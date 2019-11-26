from pathlib import Path
from typing import Union

import cv2


def get_specific_frame(path_to_video: Union[str, Path], frame_id: int):
    """
    Get specify frame from video.
    Parameters
    ----------
    path_to_video: str or Path
        Path to video file.
    frame_id: int
        Number of the desired frame in the video.

    Returns
    -------
    np.array
        Desired video frame.


    Raises
    -------
    FileExistsError
        if path_to_video file not exists.
    FrameNotFound
        if frame_id not exists in video (frame_id more then number of all frames or less then 0).

    """
    path_to_video = Path(path_to_video)
    if not path_to_video.exists():
        raise FileExistsError(f'Video {path_to_video} not exists.')
    video = cv2.VideoCapture(str(path_to_video))
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    is_success, frame = video.read()
    if is_success:
        return frame
    else:
        raise FrameNotFound(f'Frame {frame_id} not exists in video {path_to_video}')


class FrameNotFound(Exception):
    pass
