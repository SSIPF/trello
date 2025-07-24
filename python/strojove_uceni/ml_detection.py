import os
os.environ["XDG_SESSION_TYPE"] = "xcb"

import pygame
import cv2
import numpy as np
import time
from datetime import datetime
from djitellopy import Tello
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

# Constants
S = 60  # movement speed
FPS = 120
IMG_SIZE = 224

# Classes
class_names = ["group1", "group2"]

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# CNN model (same as in training)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 112x112

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 56x56

            nn.Conv2d(32, 64, 3, padding=1),  # <--- added layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 28x28
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 100),  # adjusted input features
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load trained model
model = Net()
model.load_state_dict(torch.load("../ml_sk3/drone_classifier.pth", map_location="cpu"))
model.eval()

# Create folders for saving
os.makedirs("../ml_sk3/group1", exist_ok=True)
os.makedirs("../ml_sk3/group2", exist_ok=True)

class DroneUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tello Camera Stream")
        self.screen = pygame.display.set_mode([960, 720])
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        self.tello = Tello()
        self.tello.connect()
        print(f"Battery Life: {self.tello.get_battery()}%")

        self.font = pygame.font.SysFont("Arial", 36)
        self.prediction = ""

        self.tello.set_speed(10)
        self.tello.streamoff()
        self.tello.streamon()

        self.frame_read = self.tello.get_frame_read()

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

        self.send_rc_control = False
        self.last_classification_time = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if self.frame_read.stopped:
                break

            # Get and show frame
            frame = self.frame_read.frame
            text = f"Battery: {self.tello.get_battery()}%"
            cv2.putText(frame, text, (5, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = cv2.resize(frame, (960, 720))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.rot90(frame_rgb)
            frame_rgb = np.flipud(frame_rgb)
            surface = pygame.surfarray.make_surface(frame_rgb)
            self.screen.blit(surface, (0, 0))
            if self.prediction:
                label = self.font.render(f"Predicted: {self.prediction}", True, (255, 255, 255))
                self.screen.blit(label, (10, 10))

            pygame.display.update()

            # Classify once per second
            if time.time() - self.last_classification_time >= 1:
                self.classify_frame(frame)
                self.last_classification_time = time.time()

            time.sleep(1 / FPS)

        self.tello.end()
        pygame.quit()

    def classify_frame(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(1).item()
            self.prediction = class_names[pred]
            print("Predicted:", self.prediction)

    def keydown(self, key):
        if key == pygame.K_w:
            self.for_back_velocity = S
        elif key == pygame.K_s:
            self.for_back_velocity = -S
        elif key == pygame.K_a:
            self.left_right_velocity = -S
        elif key == pygame.K_d:
            self.left_right_velocity = S
        elif key == pygame.K_UP:
            self.up_down_velocity = S
        elif key == pygame.K_DOWN:
            self.up_down_velocity = -S
        elif key == pygame.K_LEFT:
            self.yaw_velocity = -S
        elif key == pygame.K_RIGHT:
            self.yaw_velocity = S

    def keyup(self, key):
        if key in [pygame.K_w, pygame.K_s]:
            self.for_back_velocity = 0
        elif key in [pygame.K_a, pygame.K_d]:
            self.left_right_velocity = 0
        elif key in [pygame.K_UP, pygame.K_DOWN]:
            self.up_down_velocity = 0
        elif key in [pygame.K_LEFT, pygame.K_RIGHT]:
            self.yaw_velocity = 0
        elif key == pygame.K_t:
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        if self.send_rc_control:
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.for_back_velocity,
                self.up_down_velocity,
                self.yaw_velocity
            )


ui = DroneUI()
ui.run()
