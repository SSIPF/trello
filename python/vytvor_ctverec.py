from djitellopy import Tello
import time


# pripoj se k tello
tello = Tello()
tello.connect()

# baterie
print("Baterie:", tello.get_battery())

# vzlet
tello.takeoff()
time.sleep(2)

# strana čtverce
length = 100

# kreslení čtverce
for i in range(4):
    tello.move_forward(length)
    tello.rotate_clockwise(90)

# přistání
tello.land()
tello.end()
print("Hotovo :)")
