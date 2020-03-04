from pathlib import Path

import cv2

from mlutils.path import PathType, PathsType, check_path_type


class FrameNotFound(Exception):
    pass


class ImagesReader:

    def __init__(self, file_names: PathsType, read_flag=cv2.IMREAD_COLOR):
        """
        Class for iterate by images.
        Parameters
        ----------
        file_names:
            Path(s) to images for future iterate.
        read_flag:
            Flag for cv2.imread function.
        """
        self.paths = check_path_type(file_names, check_exists=True, as_list=True)
        self.read_flag = read_flag
        self._current_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._check_current_id()
        current_path = self.paths[self._current_id]
        image = cv2.imread(current_path, self.read_flag)
        self._current_id += 1
        return image

    def _check_current_id(self):
        if self._current_id == len(self.paths):
            raise StopIteration


class VideoReader:

    def __init__(self, file_names: PathsType, config=None):
        """
        Class for iterate by video frames.
        Parameters
        ----------
        file_names:
            Path to video file for future frame iterations.
        config:
            Config for cv2.set function.
        """
        self.path = check_path_type(file_names, check_exists=True, as_list=False)
        self.video = cv2.VideoCapture(self.path)
        if not self.video.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.path))

        self.config = config
        if config is not None:
            for key, value in config.items():
                self.video.set(key, value)

    def __iter__(self):
        return self

    def __next__(self):
        was_read, img = self.video.read()
        if not was_read:
            raise StopIteration
        return img


def get_specific_frame(path_to_video: PathType, frame_id: int):
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
    config_for_reader = {
        cv2.CAP_PROP_POS_FRAMES: frame_id
    }
    reader = VideoReader(path_to_video, config_for_reader)
    try:
        return next(reader)
    except StopIteration:
        raise FrameNotFound(f'Frame {frame_id} not exists in video {path_to_video}')
