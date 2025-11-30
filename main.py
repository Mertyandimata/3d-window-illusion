"""
 OFF-AXIS PROJECTION
==================================================
python main.py
"""

import pygame
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
import threading
import math
import warnings
warnings.filterwarnings("ignore")


class Config:
    WIDTH = 1280
    HEIGHT = 720
    FPS = 60
    CAMERA_ID = 0
    SENSITIVITY = 700
    SMOOTHING = 0.06
    EYE_DISTANCE = 450
    FRAME_DEPTH = 120
    ROOM_WIDTH = 640
    ROOM_HEIGHT = 360
    ROOM_DEPTH = 700
    BG_COLOR = (3, 3, 5)
    NEON_ORANGE = (255, 100, 0)
    NEON_GREEN = (0, 255, 100)
    NEON_WHITE = (255, 255, 255)
    GRID_COLOR = (25, 50, 30)
    GRID_BRIGHT = (40, 80, 45)
    CAM_SIZE = 140


class KalmanFilter2D:
    def __init__(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 0.1
        dt = 1/60
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.Q = np.eye(4) * 0.001
        self.R = np.eye(2) * 0.05
    
    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]


class HeadTracker:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.raw = np.array([0.0, 0.0])
        self.detected = False
        self.frame_mini = None
        self.running = True
        self.kalman = KalmanFilter2D()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face.process(rgb)
            
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark[168]
                self.raw = np.array([(lm.x - 0.5) * 2, (lm.y - 0.5) * 2])
                self.detected = True
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 120), 2)
                cv2.line(frame, (cx-15, cy), (cx+15, cy), (255, 255, 255), 1)
                cv2.line(frame, (cx, cy-15), (cx, cy+15), (255, 255, 255), 1)
            else:
                self.detected = False
            
            self.frame_mini = cv2.resize(frame, (Config.CAM_SIZE, int(Config.CAM_SIZE * 0.75)))
            time.sleep(1/120)
    
    def get_position(self):
        return self.kalman.update(self.raw)
    
    def get_frame(self):
        return self.frame_mini
    
    def close(self):
        self.running = False
        self.cap.release()


class Projector3D:
    def __init__(self, screen_w, screen_h):
        self.cx = screen_w // 2
        self.cy = screen_h // 2
        self.eye_z = Config.EYE_DISTANCE
    
    def project(self, obj_x, obj_y, obj_z, eye_x, eye_y):
        if obj_z <= -self.eye_z:
            obj_z = -self.eye_z + 1
        factor = self.eye_z / (self.eye_z + obj_z)
        screen_x = self.cx + (obj_x - eye_x) * factor + eye_x
        screen_y = self.cy + (obj_y - eye_y) * factor + eye_y
        return int(screen_x), int(screen_y), factor


def draw_glow_line(surface, color, start, end, width=2, glow=True):
    if glow:
        for i in range(4, 0, -1):
            glow_color = tuple(max(0, c // (i + 1)) for c in color)
            pygame.draw.line(surface, glow_color, start, end, width + i * 4)
    pygame.draw.line(surface, color, start, end, width)


def draw_glow_polygon(surface, color, points, width=0):
    if width == 0:
        pygame.draw.polygon(surface, color, points)
    else:
        for i in range(3, 0, -1):
            glow_color = tuple(max(0, c // (i + 1)) for c in color)
            pygame.draw.polygon(surface, glow_color, points, width + i * 2)
        pygame.draw.polygon(surface, color, points, width)


def draw_glow_rect(surface, color, rect, width=2):
    for i in range(4, 0, -1):
        glow_color = tuple(max(0, c // (i + 1)) for c in color)
        expanded = (rect[0] - i*2, rect[1] - i*2, rect[2] + i*4, rect[3] + i*4)
        pygame.draw.rect(surface, glow_color, expanded, width + i)
    pygame.draw.rect(surface, color, rect, width)


def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
    pygame.display.set_caption("3D YANDIMATA ILLUSION")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    font_large = pygame.font.Font(None, 28)
    
    tracker = HeadTracker()
    projector = Projector3D(Config.WIDTH, Config.HEIGHT)
    
    smoothed = np.array([0.0, 0.0])
    t = 0
    
    W, H = Config.WIDTH, Config.HEIGHT
    RW = Config.ROOM_WIDTH
    RH = Config.ROOM_HEIGHT
    RD = Config.ROOM_DEPTH
    
    # Prizma boyutları
    PRISM_WIDTH = 80
    PRISM_HEIGHT = 80
    PRISM_FRONT_Z = -100
    PRISM_BACK_Z = 650
    
    running = True
    while running:
        t += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        raw = tracker.get_position()
        smoothed = smoothed * (1 - Config.SMOOTHING) + raw * Config.SMOOTHING
        eye_x = smoothed[0] * Config.SENSITIVITY
        eye_y = smoothed[1] * Config.SENSITIVITY
        
        screen.fill(Config.BG_COLOR)
        
        # İç çerçeve köşeleri
        inner_tl = projector.project(-RW, -RH, 0, eye_x, eye_y)
        inner_tr = projector.project(RW, -RH, 0, eye_x, eye_y)
        inner_br = projector.project(RW, RH, 0, eye_x, eye_y)
        inner_bl = projector.project(-RW, RH, 0, eye_x, eye_y)
        
        # ==================== ZEMİN ====================
        floor_lines = 12
        for i in range(floor_lines + 1):
            z = RD * i / floor_lines
            px1, py1, _ = projector.project(-RW, RH, z, eye_x, eye_y)
            px2, py2, _ = projector.project(RW, RH, z, eye_x, eye_y)
            col = Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1)
        
        for i in range(13):
            x = -RW + (2 * RW * i / 12)
            px1, py1, _ = projector.project(x, RH, 0, eye_x, eye_y)
            px2, py2, _ = projector.project(x, RH, RD, eye_x, eye_y)
            col = Config.NEON_GREEN if i == 6 else (Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR)
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1 if i != 6 else 2)
        
        # ==================== TAVAN ====================
        for i in range(floor_lines + 1):
            z = RD * i / floor_lines
            px1, py1, _ = projector.project(-RW, -RH, z, eye_x, eye_y)
            px2, py2, _ = projector.project(RW, -RH, z, eye_x, eye_y)
            col = Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1)
        
        for i in range(13):
            x = -RW + (2 * RW * i / 12)
            px1, py1, _ = projector.project(x, -RH, 0, eye_x, eye_y)
            px2, py2, _ = projector.project(x, -RH, RD, eye_x, eye_y)
            col = Config.NEON_GREEN if i == 6 else (Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR)
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1 if i != 6 else 2)
        
        # ==================== SOL DUVAR ====================
        for i in range(floor_lines + 1):
            z = RD * i / floor_lines
            px1, py1, _ = projector.project(-RW, -RH, z, eye_x, eye_y)
            px2, py2, _ = projector.project(-RW, RH, z, eye_x, eye_y)
            col = Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1)
        
        for i in range(9):
            y = -RH + (2 * RH * i / 8)
            px1, py1, _ = projector.project(-RW, y, 0, eye_x, eye_y)
            px2, py2, _ = projector.project(-RW, y, RD, eye_x, eye_y)
            col = Config.NEON_ORANGE if i == 4 else (Config.GRID_BRIGHT if i % 2 == 0 else Config.GRID_COLOR)
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1 if i != 4 else 2)
        
        # ==================== SAĞ DUVAR ====================
        for i in range(floor_lines + 1):
            z = RD * i / floor_lines
            px1, py1, _ = projector.project(RW, -RH, z, eye_x, eye_y)
            px2, py2, _ = projector.project(RW, RH, z, eye_x, eye_y)
            col = Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1)
        
        for i in range(9):
            y = -RH + (2 * RH * i / 8)
            px1, py1, _ = projector.project(RW, y, 0, eye_x, eye_y)
            px2, py2, _ = projector.project(RW, y, RD, eye_x, eye_y)
            col = Config.NEON_ORANGE if i == 4 else (Config.GRID_BRIGHT if i % 2 == 0 else Config.GRID_COLOR)
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1 if i != 4 else 2)
        
        # ==================== ARKA DUVAR ====================
        for i in range(13):
            x = -RW + (2 * RW * i / 12)
            px1, py1, _ = projector.project(x, -RH, RD, eye_x, eye_y)
            px2, py2, _ = projector.project(x, RH, RD, eye_x, eye_y)
            col = Config.NEON_GREEN if i == 6 else (Config.GRID_BRIGHT if i % 3 == 0 else Config.GRID_COLOR)
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1 if i != 6 else 2)
        
        for i in range(9):
            y = -RH + (2 * RH * i / 8)
            px1, py1, _ = projector.project(-RW, y, RD, eye_x, eye_y)
            px2, py2, _ = projector.project(RW, y, RD, eye_x, eye_y)
            col = Config.NEON_ORANGE if i == 4 else (Config.GRID_BRIGHT if i % 2 == 0 else Config.GRID_COLOR)
            pygame.draw.line(screen, col, (px1, py1), (px2, py2), 1 if i != 4 else 2)
        
        # Arka duvar çerçevesi
        back_corners = [
            projector.project(-RW, -RH, RD, eye_x, eye_y),
            projector.project(RW, -RH, RD, eye_x, eye_y),
            projector.project(RW, RH, RD, eye_x, eye_y),
            projector.project(-RW, RH, RD, eye_x, eye_y)
        ]
        back_pts = [(p[0], p[1]) for p in back_corners]
        draw_glow_polygon(screen, Config.NEON_GREEN, back_pts, 2)
        
        # Oda köşeleri
        corners_back = [(-RW, -RH, RD), (RW, -RH, RD), (RW, RH, RD), (-RW, RH, RD)]
        inner_corners = [inner_tl, inner_tr, inner_br, inner_bl]
        for i in range(4):
            pb = projector.project(*corners_back[i], eye_x, eye_y)
            ic = inner_corners[i]
            draw_glow_line(screen, Config.NEON_ORANGE, (int(ic[0]), int(ic[1])), (pb[0], pb[1]), 2, glow=False)
        
        # ==================== PARÇACIKLAR ====================
        for i in range(35):
            pz = 80 + (i * 47) % (RD - 150)
            px = -RW + 80 + (i * 137) % (RW * 2 - 160)
            py = -RH + 80 + (i * 89) % (RH * 2 - 160)
            px += math.sin(t * 0.02 + i) * 20
            py += math.cos(t * 0.015 + i * 0.7) * 15
            sx, sy, factor = projector.project(px, py, pz, eye_x, eye_y)
            size = max(2, int(4 * factor))
            brightness = int(150 + 100 * factor)
            if i % 3 == 0:
                color = (brightness, int(brightness * 0.5), 0)
            elif i % 3 == 1:
                color = (0, brightness, int(brightness * 0.5))
            else:
                color = (brightness // 2, brightness // 2, brightness // 2)
            pygame.draw.circle(screen, color, (sx, sy), size)
        
        # ==================== 3D PRİZMA ====================
        float_y = math.sin(t * 0.03) * 8
        float_x = math.cos(t * 0.02) * 5
        
        # Ön yüz köşeleri
        front_tl = (-PRISM_WIDTH + float_x, -PRISM_HEIGHT + float_y, PRISM_FRONT_Z)
        front_tr = (PRISM_WIDTH + float_x, -PRISM_HEIGHT + float_y, PRISM_FRONT_Z)
        front_br = (PRISM_WIDTH + float_x, PRISM_HEIGHT + float_y, PRISM_FRONT_Z)
        front_bl = (-PRISM_WIDTH + float_x, PRISM_HEIGHT + float_y, PRISM_FRONT_Z)
        
        # Arka yüz köşeleri
        back_tl = (-PRISM_WIDTH + float_x * 0.3, -PRISM_HEIGHT + float_y * 0.3, PRISM_BACK_Z)
        back_tr = (PRISM_WIDTH + float_x * 0.3, -PRISM_HEIGHT + float_y * 0.3, PRISM_BACK_Z)
        back_br = (PRISM_WIDTH + float_x * 0.3, PRISM_HEIGHT + float_y * 0.3, PRISM_BACK_Z)
        back_bl = (-PRISM_WIDTH + float_x * 0.3, PRISM_HEIGHT + float_y * 0.3, PRISM_BACK_Z)
        
        # Project corners
        f_tl = projector.project(*front_tl, eye_x, eye_y)
        f_tr = projector.project(*front_tr, eye_x, eye_y)
        f_br = projector.project(*front_br, eye_x, eye_y)
        f_bl = projector.project(*front_bl, eye_x, eye_y)
        b_tl = projector.project(*back_tl, eye_x, eye_y)
        b_tr = projector.project(*back_tr, eye_x, eye_y)
        b_br = projector.project(*back_br, eye_x, eye_y)
        b_bl = projector.project(*back_bl, eye_x, eye_y)
        
        # Yüzler
        faces = []
        faces.append(([(b_tl[0], b_tl[1]), (b_tr[0], b_tr[1]), (b_br[0], b_br[1]), (b_bl[0], b_bl[1])], (15, 35, 20), PRISM_BACK_Z))
        if eye_x > 0:
            faces.append(([(b_tl[0], b_tl[1]), (f_tl[0], f_tl[1]), (f_bl[0], f_bl[1]), (b_bl[0], b_bl[1])], (25, 55, 30), 0))
        if eye_x < 0:
            faces.append(([(f_tr[0], f_tr[1]), (b_tr[0], b_tr[1]), (b_br[0], b_br[1]), (f_br[0], f_br[1])], (25, 55, 30), 0))
        if eye_y > 0:
            faces.append(([(b_tl[0], b_tl[1]), (b_tr[0], b_tr[1]), (f_tr[0], f_tr[1]), (f_tl[0], f_tl[1])], (35, 75, 40), 0))
        if eye_y < 0:
            faces.append(([(f_bl[0], f_bl[1]), (f_br[0], f_br[1]), (b_br[0], b_br[1]), (b_bl[0], b_bl[1])], (20, 45, 25), 0))
        faces.append(([(f_tl[0], f_tl[1]), (f_tr[0], f_tr[1]), (f_br[0], f_br[1]), (f_bl[0], f_bl[1])], (50, 110, 60), PRISM_FRONT_Z))
        
        faces.sort(key=lambda f: f[2], reverse=True)
        for pts, col, _ in faces:
            pygame.draw.polygon(screen, col, pts)
        
        # Ön yüz MAT - solid dolgu (içi görünmez)
        front_face_pts = [(f_tl[0], f_tl[1]), (f_tr[0], f_tr[1]), (f_br[0], f_br[1]), (f_bl[0], f_bl[1])]
        pygame.draw.polygon(screen, (20, 45, 25), front_face_pts)  # Koyu yeşil mat yüzey
        
        # Neon kenarlar
        pulse = 0.8 + 0.2 * math.sin(t * 0.15)
        edge_color = tuple(int(c * pulse) for c in Config.NEON_ORANGE)
        
        front_edges = [((f_tl[0], f_tl[1]), (f_tr[0], f_tr[1])), ((f_tr[0], f_tr[1]), (f_br[0], f_br[1])),
                       ((f_br[0], f_br[1]), (f_bl[0], f_bl[1])), ((f_bl[0], f_bl[1]), (f_tl[0], f_tl[1]))]
        for start, end in front_edges:
            for g in range(6, 0, -1):
                glow_col = tuple(max(0, c // (g + 1)) for c in edge_color)
                pygame.draw.line(screen, glow_col, start, end, 4 + g * 3)
            pygame.draw.line(screen, edge_color, start, end, 4)
            pygame.draw.line(screen, Config.NEON_WHITE, start, end, 2)
        
        conn_color = tuple(int(c * pulse * 0.7) for c in Config.NEON_GREEN)
        connections = [((f_tl[0], f_tl[1]), (b_tl[0], b_tl[1])), ((f_tr[0], f_tr[1]), (b_tr[0], b_tr[1])),
                       ((f_br[0], f_br[1]), (b_br[0], b_br[1])), ((f_bl[0], f_bl[1]), (b_bl[0], b_bl[1]))]
        for start, end in connections:
            for g in range(4, 0, -1):
                glow_col = tuple(max(0, c // (g + 1)) for c in conn_color)
                pygame.draw.line(screen, glow_col, start, end, 2 + g * 2)
            pygame.draw.line(screen, conn_color, start, end, 2)
        
        # Tünel ara çizgileri
        for i in range(1, 8):
            ratio = i / 8
            z = PRISM_FRONT_Z + (PRISM_BACK_Z - PRISM_FRONT_Z) * ratio
            fx = float_x * (1 - ratio * 0.7)
            fy = float_y * (1 - ratio * 0.7)
            seg_tl = projector.project(-PRISM_WIDTH + fx, -PRISM_HEIGHT + fy, z, eye_x, eye_y)
            seg_tr = projector.project(PRISM_WIDTH + fx, -PRISM_HEIGHT + fy, z, eye_x, eye_y)
            seg_br = projector.project(PRISM_WIDTH + fx, PRISM_HEIGHT + fy, z, eye_x, eye_y)
            seg_bl = projector.project(-PRISM_WIDTH + fx, PRISM_HEIGHT + fy, z, eye_x, eye_y)
            seg_color = tuple(int(c * (1 - ratio * 0.6)) for c in Config.NEON_GREEN)
            pygame.draw.line(screen, seg_color, (seg_tl[0], seg_tl[1]), (seg_tr[0], seg_tr[1]), 1)
            pygame.draw.line(screen, seg_color, (seg_tr[0], seg_tr[1]), (seg_br[0], seg_br[1]), 1)
            pygame.draw.line(screen, seg_color, (seg_br[0], seg_br[1]), (seg_bl[0], seg_bl[1]), 1)
            pygame.draw.line(screen, seg_color, (seg_bl[0], seg_bl[1]), (seg_tl[0], seg_tl[1]), 1)
        
        back_color = tuple(int(c * 0.4) for c in Config.NEON_GREEN)
        back_edges = [((b_tl[0], b_tl[1]), (b_tr[0], b_tr[1])), ((b_tr[0], b_tr[1]), (b_br[0], b_br[1])),
                      ((b_br[0], b_br[1]), (b_bl[0], b_bl[1])), ((b_bl[0], b_bl[1]), (b_tl[0], b_tl[1]))]
        for start, end in back_edges:
            pygame.draw.line(screen, back_color, start, end, 2)
        
        # ==================== ÇERÇEVE ====================
        outer_tl, outer_tr, outer_br, outer_bl = (0, 0), (W, 0), (W, H), (0, H)
        
        top_pts = [outer_tl, outer_tr, (inner_tr[0], inner_tr[1]), (inner_tl[0], inner_tl[1])]
        pygame.draw.polygon(screen, (12, 20, 15), top_pts)
        for i in range(8):
            x = inner_tl[0] + (inner_tr[0] - inner_tl[0]) * i / 7
            x_out = outer_tl[0] + (outer_tr[0] - outer_tl[0]) * i / 7
            pygame.draw.line(screen, Config.GRID_COLOR, (int(x_out), outer_tl[1]), (int(x), int(inner_tl[1])), 1)
        
        bot_pts = [(inner_bl[0], inner_bl[1]), (inner_br[0], inner_br[1]), outer_br, outer_bl]
        pygame.draw.polygon(screen, (12, 20, 15), bot_pts)
        for i in range(8):
            x = inner_bl[0] + (inner_br[0] - inner_bl[0]) * i / 7
            x_out = outer_bl[0] + (outer_br[0] - outer_bl[0]) * i / 7
            pygame.draw.line(screen, Config.GRID_COLOR, (int(x), int(inner_bl[1])), (int(x_out), outer_bl[1]), 1)
        
        left_pts = [outer_tl, (inner_tl[0], inner_tl[1]), (inner_bl[0], inner_bl[1]), outer_bl]
        pygame.draw.polygon(screen, (15, 25, 18), left_pts)
        for i in range(6):
            y = inner_tl[1] + (inner_bl[1] - inner_tl[1]) * i / 5
            y_out = outer_tl[1] + (outer_bl[1] - outer_tl[1]) * i / 5
            pygame.draw.line(screen, Config.GRID_COLOR, (outer_tl[0], int(y_out)), (int(inner_tl[0]), int(y)), 1)
        
        right_pts = [(inner_tr[0], inner_tr[1]), outer_tr, outer_br, (inner_br[0], inner_br[1])]
        pygame.draw.polygon(screen, (15, 25, 18), right_pts)
        for i in range(6):
            y = inner_tr[1] + (inner_br[1] - inner_tr[1]) * i / 5
            y_out = outer_tr[1] + (outer_br[1] - outer_tr[1]) * i / 5
            pygame.draw.line(screen, Config.GRID_COLOR, (int(inner_tr[0]), int(y)), (outer_tr[0], int(y_out)), 1)
        
        inner_frame = [(inner_tl[0], inner_tl[1]), (inner_tr[0], inner_tr[1]), (inner_br[0], inner_br[1]), (inner_bl[0], inner_bl[1])]
        draw_glow_polygon(screen, Config.NEON_ORANGE, inner_frame, 3)
        pygame.draw.rect(screen, Config.NEON_ORANGE, (0, 0, W, H), 4)
        
        # ==================== HUD (minimal) ====================
        status_col = Config.NEON_GREEN if tracker.detected else Config.NEON_ORANGE
        if tracker.detected:
            pygame.draw.circle(screen, status_col, (W - 25, 25), 6 + int(2 * math.sin(t * 0.2)))
        else:
            pygame.draw.circle(screen, status_col, (W - 25, 25), 6)
        
        screen.blit(font.render(f"{int(clock.get_fps())}fps", True, (50, 50, 50)), (W - 55, H - 22))
        
        # ==================== KAMERA ====================
        cam_w, cam_h = Config.CAM_SIZE, int(Config.CAM_SIZE * 0.75)
        cam_x, cam_y = 15, H - cam_h - 15
        frame = tracker.get_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(surf, (cam_x, cam_y))
        border_col = Config.NEON_GREEN if tracker.detected else Config.NEON_ORANGE
        pygame.draw.rect(screen, border_col, (cam_x, cam_y, cam_w, cam_h), 2)
        if t % 50 < 25:
            pygame.draw.circle(screen, (255, 40, 40), (cam_x + cam_w - 10, cam_y + 10), 4)
        
        pygame.display.flip()
        clock.tick(Config.FPS)
    
    tracker.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()