# 🎯 Parallax 3D Head Tracking

Real-time 3D parallax illusion using webcam face tracking. Turn your 2D screen into a window to a 3D world.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-red.svg)



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


- LinkedIn: https://www.linkedin.com/in/mert-yandimata/?originalSubdomain=it

---

⭐ If you found this project interesting, please consider giving it a star!
