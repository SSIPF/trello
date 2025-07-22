from djitellopy import Tello
import cv2
import time


# název fotky
FILENAME = "img_drone_1.jpg"


# připoj s k Tello
tello = Tello()
tello.connect()
tello.streamon()

# čekej 2 s
time.sleep(2)

# načti obrázek a ulož jej
frame = tello.get_frame_read().frame
cv2.imwrite(FILENAME, frame)
print(f"Photo saved as {FILENAME}")

tello.streamoff()
tello.end()
