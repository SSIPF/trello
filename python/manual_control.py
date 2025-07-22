from djitellopy import tello
import KeyPressModule as kp
import time
import cv2

kp.init()
Tello = tello.Tello()
Tello.connect()
print(Tello.get_battery())

global img

Tello.streamon()


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0

    speed = 50

    if kp.getKey("LEFT"):
        lr = -speed

    elif kp.getKey("RIGHT"):
        lr = speed

    if kp.getKey("UP"):
        fb = speed

    elif kp.getKey("DOWN"):
        fb = -speed

    if kp.getKey("w"):
        ud = speed

    elif kp.getKey("s"):
        ud = -speed

    if kp.getKey("a"):
        yv = -speed

    elif kp.getKey("d"):
        yv = speed

    if kp.getKey("q"):
        Tello.land()
        time.sleep(3)

    if kp.getKey("e"):
        Tello.takeoff()

    if kp.getKey("z"):
        cv2.imwrite(f'Resources/Images/{time.time()}.jpg', img)
        time.sleep(0.3)

    return [lr, fb, ud, yv]


while True:
    vals = getKeyboardInput()

    Tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    img = Tello.get_frame_read().frame

    img = cv2.resize(img, (360, 240))

    cv2.imshow("Image", img)

    cv2.waitKey(1)
