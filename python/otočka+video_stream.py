from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()
print(f"Baterie: {tello.get_battery()}%")

tello.streamon()  # zapnout video stream

# Odstartuj a udělej otočku
tello.takeoff()
tello.rotate_clockwise(360)

# Zobrazuj video 5 sekund
start_time = time.time()
while time.time() - start_time < 5:
    frame = tello.get_frame_read().frame
    cv2.imshow("Tello Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.land()
tello.end()
cv2.destroyAllWindows()
