# Virtual AR Piano 

A virtual piano that uses your phone camera to detect finger movements and play notes in real-time.

[![Watch the demo](https://img.youtube.com/vi/0t2wSjs-yuU/0.jpg)](https://youtu.be/0t2wSjs-yuU)

## Project Overview

This project creates a virtual piano interface using computer vision.

## Key Features

- **Real-time hand tracking** using MediaPipe
- **Depth-aware key detection** using MiDaS
- **Multi-finger support** for playing chords
- **Visual feedback** with colored finger indicators


| Uses | Purpose |
|------------|---------|
| **MediaPipe** | Real-time hand landmark detection |
| **MiDaS** | Depth estimation from single images |
| **OpenCV** | Video processing and interface rendering |
| **Python** | Main programming language |


## Project Structure

```
virtual-piano/
├── main.py              # Main application
├── virtual_keyboard.py  # Keyboard rendering class
├── key_tracker.py       # Key state tracking
└── README.md           # This file
```

## Technical Details

- Uses 21 hand landmarks from MediaPipe
- Depth estimation with MiDaS DPT Large model
- Real-time processing at 30 FPS
- Configurable depth thresholds for key presses
