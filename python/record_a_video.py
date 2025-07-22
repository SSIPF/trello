from djitellopy import Tello
import cv2
import time


FILENAME = "vid_drone_1.jpg"    # název
VIDEO_TIME = 10                 # délka v s

# připoj s k Tello
tello = Tello()
tello.connect()

# začni natáčení
tello.streamon()
frame_read = tello.get_frame_read()

video = cv2.VideoWriter(
    FILENAME,
    cv2.VideoWriter_fourcc(*"XVID"),
    30,
    (960, 720)
)

# nahrávej
start_time = time.time()
while time.time() - start_time < VIDEO_TIME:
    frame = frame_read.frame
    if frame is not None:
        resized_frame = cv2.resize(frame, (960, 720))
        video.write(resized_frame)

# konec
video.release()
tello.streamoff()
tello.end()
