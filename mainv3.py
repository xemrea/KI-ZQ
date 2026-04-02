import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import pygame
import math
import time
from enum import Enum

# Pygame Initialisierung
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🎯 Waste Sorting Master - Stickman Edition")
clock = pygame.time.Clock()
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)

# Farben
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (100, 150, 255)
SKIN = (255, 200, 100)

class GameState(Enum):
    IDLE = 1
    DETECTING = 2
    THROWING = 3
    RESULT = 4

class Stickman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.throw_progress = 0
        self.throwing = False
        self.throw_target_x = WIDTH - 150
        self.throw_target_y = HEIGHT - 150
        
    def draw(self, surface, throwing=False, progress=0):
        # Kopf
        head_x, head_y = self.x, self.y - 60
        pygame.draw.circle(surface, SKIN, (int(head_x), int(head_y)), 20)
        
        # Augen
        pygame.draw.circle(surface, BLACK, (int(head_x - 8), int(head_y - 8)), 3)
        pygame.draw.circle(surface, BLACK, (int(head_x + 8), int(head_y - 8)), 3)
        
        # Mund
        pygame.draw.line(surface, BLACK, (int(head_x - 5), int(head_y + 5)), (int(head_x + 5), int(head_y + 5)), 2)
        
        # Körper
        body_y_start = head_y + 20
        body_y_end = head_y + 60
        pygame.draw.line(surface, BLACK, (int(head_x), int(body_y_start)), (int(head_x), int(body_y_end)), 3)
        
        if throwing:
            # Wurfpose
            arm_left_x = head_x - 30 - (progress * 50)
            arm_left_y = body_y_start + 20 - (progress * 40)
            arm_right_x = head_x + 30 - (progress * 20)
            arm_right_y = body_y_start + 20
            
            # Beine
            leg_left_x = head_x - 10
            leg_right_x = head_x + 10
            leg_y = body_y_end
        else:
            # Normale Pose
            arm_left_x = head_x - 30
            arm_left_y = body_y_start + 20
            arm_right_x = head_x + 30
            arm_right_y = body_y_start + 20
            
            # Beine
            leg_left_x = head_x - 10
            leg_right_x = head_x + 10
            leg_y = body_y_end
        
        # Arme
        pygame.draw.line(surface, BLACK, (int(head_x), int(body_y_start + 20)), (int(arm_left_x), int(arm_left_y)), 3)
        pygame.draw.line(surface, BLACK, (int(head_x), int(body_y_start + 20)), (int(arm_right_x), int(arm_right_y)), 3)
        
        # Beine
        pygame.draw.line(surface, BLACK, (int(leg_left_x), int(leg_y)), (int(leg_left_x), int(leg_y + 40)), 3)
        pygame.draw.line(surface, BLACK, (int(leg_right_x), int(leg_y)), (int(leg_right_x), int(leg_y + 40)), 3)

class Trash:
    def __init__(self):
        self.x = WIDTH - 150
        self.y = HEIGHT - 150
        self.lid_angle = 0
        
    def draw(self, surface):
        # Mülleimer Körper
        trash_width = 100
        trash_height = 120
        trash_rect = pygame.Rect(self.x - trash_width // 2, self.y - trash_height, trash_width, trash_height)
        pygame.draw.rect(surface, GRAY, trash_rect, 0)
        pygame.draw.rect(surface, BLACK, trash_rect, 3)
        
        # Deckel
        lid_x = self.x
        lid_y = self.y - trash_height
        lid_width = 110
        
        # Deckel animieren wenn offen
        if self.lid_angle > 0:
            self.lid_angle = min(self.lid_angle + 15, 90)
        
        # Deckel zeichnen (einfaches Rechteck mit Rotation simulieren)
        if self.lid_angle > 0:
            lid_offset = math.sin(math.radians(self.lid_angle)) * 20
            pygame.draw.rect(surface, (150, 150, 150), 
                           (self.x - lid_width // 2, self.y - trash_height - 10 - lid_offset, lid_width, 15), 0)
        else:
            pygame.draw.rect(surface, (150, 150, 150), 
                           (self.x - lid_width // 2, self.y - trash_height - 10, lid_width, 15), 0)
        
        pygame.draw.rect(surface, BLACK, 
                        (self.x - lid_width // 2, self.y - trash_height - 10 - (math.sin(math.radians(self.lid_angle)) * 20), 
                         lid_width, 15), 2)
        
    def open_lid(self):
        self.lid_angle = 0

class Projectile:
    def __init__(self, start_x, start_y, target_x, target_y, label, duration=1.0):
        self.start_x = start_x
        self.start_y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.progress = 0
        self.max_duration = duration
        self.label = label
        self.active = True
        
    def update(self, dt):
        self.progress += dt / self.max_duration
        if self.progress >= 1.0:
            self.active = False
            
    def get_position(self):
        # Parabolische Flugbahn
        t = self.progress
        x = self.start_x + (self.target_x - self.start_x) * t
        y = self.start_y + (self.target_y - self.start_y) * t - (t * (1 - t) * 150)
        return x, y
    
    def draw(self, surface):
        if self.active:
            x, y = self.get_position()
            # Objekt zeichnen (Kreis mit Label)
            pygame.draw.circle(surface, YELLOW, (int(x), int(y)), 15)
            pygame.draw.circle(surface, BLACK, (int(x), int(y)), 15, 2)
            
            # Label auf Objekt
            label_text = font_small.render(self.label[:3], True, BLACK)
            surface.blit(label_text, (int(x - 10), int(y - 8)))

class SpeechBubble:
    def __init__(self, text, x, y, duration=2.0):
        self.text = text
        self.x = x
        self.y = y
        self.duration = duration
        self.elapsed = 0
        self.active = True
        
    def update(self, dt):
        self.elapsed += dt
        if self.elapsed >= self.duration:
            self.active = False
            
    def draw(self, surface):
        if self.active:
            # Sprechblase
            bubble_width = 200
            bubble_height = 80
            bubble_x = self.x - bubble_width // 2
            bubble_y = self.y - bubble_height - 20
            
            # Hintergrund
            pygame.draw.rect(surface, WHITE, (bubble_x, bubble_y, bubble_width, bubble_height))
            pygame.draw.rect(surface, BLACK, (bubble_x, bubble_y, bubble_width, bubble_height), 3)
            
            # Schwanz (Dreieck)
            pygame.draw.polygon(surface, WHITE, [
                (self.x - 10, bubble_y + bubble_height),
                (self.x + 10, bubble_y + bubble_height),
                (self.x, bubble_y + bubble_height + 15)
            ])
            pygame.draw.polygon(surface, BLACK, [
                (self.x - 10, bubble_y + bubble_height),
                (self.x + 10, bubble_y + bubble_height),
                (self.x, bubble_y + bubble_height + 15)
            ], 2)
            
            # Text
            text_surface = font_medium.render(self.text, True, BLACK)
            text_rect = text_surface.get_rect(center=(bubble_x + bubble_width // 2, bubble_y + bubble_height // 2))
            surface.blit(text_surface, text_rect)

# Modell laden
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Spiel initialisieren
stickman = Stickman(150, HEIGHT - 100)
trash = Trash()
game_state = GameState.IDLE
projectile = None
speech_bubble = None
confidence = 0
detected_class = "Warte..."
last_detection_time = 0
detection_cooldown = 2.0

# Webcam
cap = cv2.VideoCapture(0)

running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time in Sekunden
    
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_state = GameState.DETECTING
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Webcam Frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)  # Horizontal flippen
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
    else:
        frame_surface = pygame.Surface((WIDTH, HEIGHT))
        frame_surface.fill(GRAY)
    
    # Hintergrund
    screen.blit(frame_surface, (0, 0))
    
    # Overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(100)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    # Game Logic
    if game_state == GameState.DETECTING:
        if ret:
            # Klassifizierung
            frame_rgb_pil = Image.fromarray(frame_rgb)
            frame_rgb_pil = ImageOps.fit(frame_rgb_pil, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(frame_rgb_pil)
            normalized = (image_array.astype(np.float32) / 127.5) - 1
            
            interpreter.set_tensor(input_details[0]['index'], 
                                  np.expand_dims(normalized, axis=0).astype(np.float32))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            index = np.argmax(output_data)
            detected_class = class_names[index].strip()
            confidence = output_data[0][index]
            
            # Animation starten
            projectile = Projectile(stickman.x, stickman.y - 100, trash.x, trash.y - 120, detected_class, duration=1.2)
            speech_bubble = SpeechBubble(detected_class, trash.x, trash.y - 150, duration=2.0)
            trash.open_lid()
            
            game_state = GameState.THROWING
            last_detection_time = time.time()
    
    if game_state == GameState.THROWING:
        if projectile and projectile.active:
            projectile.update(dt)
        else:
            game_state = GameState.RESULT
    
    if game_state == GameState.RESULT:
        if time.time() - last_detection_time > 3.0:
            game_state = GameState.IDLE
    
    # Update Speech Bubble
    if speech_bubble:
        speech_bubble.update(dt)
    
    # Draw Stickman
    is_throwing = game_state in [GameState.THROWING, GameState.RESULT]
    throw_progress = projectile.progress if projectile else 0
    stickman.draw(screen, throwing=is_throwing, progress=throw_progress)
    
    # Draw Trash
    trash.draw(screen)
    
    # Draw Projectile
    if projectile:
        projectile.draw(screen)
    
    # Draw Speech Bubble
    if speech_bubble and speech_bubble.active:
        speech_bubble.draw(screen)
    
    # UI
    if game_state == GameState.IDLE:
        title_text = font_large.render("🎯 WASTE SORTING MASTER 🎯", True, YELLOW)
        title_rect = title_text.get_rect(center=(WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        instruction_text = font_medium.render("Drücke SPACE zum Scannen!", True, GREEN)
        instruction_rect = instruction_text.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        screen.blit(instruction_text, instruction_rect)
    
    if game_state == GameState.DETECTING:
        detecting_text = font_medium.render("📸 SCANNEN...", True, YELLOW)
        detecting_rect = detecting_text.get_rect(center=(WIDTH // 2, 50))
        screen.blit(detecting_text, detecting_rect)
    
    if game_state == GameState.RESULT:
        result_text = font_medium.render(f"✅ {detected_class} ({confidence:.1%})", True, GREEN)
        result_rect = result_text.get_rect(center=(WIDTH // 2, 50))
        screen.blit(result_text, result_rect)
    
    pygame.display.flip()

cap.release()
pygame.quit()