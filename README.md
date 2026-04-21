# AI Face Skin Retouching

This project is a completely revamped, Python-centric AI skin retouching application. It replaces legacy OpenCV Haar Cascades with Google's modern **MediaPipe FaceLandmarker Tasks API**, achieving extremely precise 3D facial mesh generation for highly targeted, natural-looking skin retouching.

## Features
- **High-Precision AI Targeting:** Leverages MediaPipe to accurately pinpoint the face and avoid smudging details like hair or background elements.
- **Frequency Separation Algorithm:** Implements an advanced bilateral filter and high-pass separation logic, akin to professional Photoshop techniques.
- **Cross-Platform & Lightweight:** Fully written in Python, meaning it runs gracefully on macOS, Windows, and Linux without the pain of compiling C++ binaries.

## Installation

Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone https://github.com/spinnovation/skinretouch.git
cd skinretouch

# Setup a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: The required `face_landmarker.task` model file is included in the project.)*

## Usage

Simply run the script via CLI. By default, it will attempt to process `1.jpeg` or `1.png` and output out to `result.png`.

```bash
# Basic usage
python main.py

# Process a specific image
python main.py my_photo.jpg

# Process and specify an output filename
python main.py my_photo.jpg --output my_retouched_photo.png
```

## How It Works

1. **Face Mesh Detection:** Identifies up to 478 3D landmarks on the subjects' faces.
2. **Skin Masking:** Creates a precise convex hull mask covering the targeted face dimensions.
3. **Bilateral Bluring & Detail High-Pass:** Smoothes the color frequencies while separating the pore/texture anomalies, and then blends them back using 50% opacity onto the original masked region.