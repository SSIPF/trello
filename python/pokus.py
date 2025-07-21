from djitellopy import Tello
import time

# Inicializace dronu
tello = Tello()

# Připojení k dronu
print("Připojuji se k dronu...")
tello.connect()

# Stav baterie
print(f"Baterie: {tello.get_battery()}%")

# Odstartování
print("Start...")
tello.takeoff()

# Let dopředu o 50 cm
print("Letím dopředu...")
tello.move_forward(50)

# Počkej 2 sekundy
time.sleep(2)

# Přistání
print("Přistávám...")
tello.land()

# Odpojení
tello.end()
