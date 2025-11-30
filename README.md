# 🎯 Parallax 3D Head Tracking

Real-time 3D parallax illusion using webcam face tracking. Turn your 2D screen into a window to a 3D world.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-red.svg)

https://github.com/user-attachments/assets/YOUR_VIDEO_ID

## ✨ Features

- **Real-time face tracking** using MediaPipe Face Mesh
- **Off-axis projection** for true 3D parallax effect
- **60 FPS** smooth rendering with Pygame
- **Kalman filtering** for stable head position tracking
- **Neon cyberpunk aesthetics** with glow effects
- **Window frame effect** - screen edges act as a physical window frame

## 🎬 How It Works

The illusion is based on **off-axis projection**, the same technique used in VR headsets and CAVE systems. By tracking your head position via webcam, the scene perspective updates in real-time, creating the illusion that your screen is a window into a 3D space.

```
Your Head Position → Webcam → MediaPipe → Kalman Filter → Off-Axis Projection → 3D Illusion
```

### The Math Behind It

```python
# Off-axis projection formula
factor = eye_distance / (eye_distance + object_z)
screen_x = center_x + (object_x - eye_x) * factor + eye_x
screen_y = center_y + (object_y - eye_y) * factor + eye_y
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Good lighting for face tracking

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/parallax-3d-head-tracking.git
cd parallax-3d-head-tracking

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| Move head | Control 3D perspective |

## 📦 Dependencies

```
pygame>=2.1.0
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.21.0
```

## 🎨 Customization

Edit `Config` class in `main.py`:

```python
class Config:
    WIDTH = 1280              # Window width
    HEIGHT = 720              # Window height
    SENSITIVITY = 700         # Head movement sensitivity
    SMOOTHING = 0.06          # Lower = more responsive
    EYE_DISTANCE = 450        # Virtual eye-to-screen distance
    
    # Colors (RGB)
    NEON_ORANGE = (255, 100, 0)
    NEON_GREEN = (0, 255, 100)
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Face not detected | Improve lighting, face the camera directly |
| Laggy movement | Close other applications, reduce `SENSITIVITY` |
| Wrong camera | Change `CAMERA_ID` in Config (0, 1, 2...) |

## 📚 References & Inspiration

- [Johnny Lee's Wii Remote Head Tracking](http://johnnylee.net/projects/wii/) - Original inspiration
- [Shopify's WonkaVision](https://shopify.github.io/spatial-commerce-projects/WonkaVision/) - Browser implementation
- [Off-Axis Projection Paper](http://160592857366.free.fr/joe/ebooks/ShareData/Generalized%20Perspective%20Projection.pdf) - Mathematical foundation

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN)

---

⭐ If you found this project interesting, please consider giving it a star!