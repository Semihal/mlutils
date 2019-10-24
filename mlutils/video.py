from pathlib import Path
import cv2


def get_specific_frame(path_to_video: Path, frame_id: int):
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
