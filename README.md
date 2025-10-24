# 🪄 OpenCV Augmented Reality Demo

This project demonstrates a **real-time Augmented Reality (AR)** overlay system using **OpenCV** and **ArUco markers**.  
It lets you project any image or video onto a flat surface marked by 4 ArUco markers visible in a webcam feed.

Three main components are included:

1. 🧩 **Basic AR Demo** – simple image overlay using marker detection.  
2. ⚡ **Enhanced AR Demo (Stabilized + Performance Overlay)** – adds smoothing, FPS counter, and timing HUD.
3. 🎯 **ArUco Marker Generator** – generate printable ArUco markers in PDF or JPG format.

---

## 🚀 Features

| Feature | Basic Version | Enhanced Version | Marker Generator |
|----------|----------------|------------------|------------------|
| ArUco marker detection | ✅ | ✅ | ✅ |
| Image overlay with perspective warp | ✅ | ✅ | ❌ |
| Real-time webcam capture | ✅ | ✅ | ❌ |
| Stabilization (smoothing) | ❌ | ✅ | ❌ |
| FPS counter | ❌ | ✅ | ❌ |
| Detection & overlay timing | ❌ | ✅ | ❌ |
| Adjustable resolution | ❌ | ✅ | ❌ |
| Clean performance overlay (HUD) | ❌ | ✅ | ❌ |
| Generate single/multiple markers | ❌ | ❌ | ✅ |
| PDF output (print-ready) | ❌ | ❌ | ✅ |
| JPG output (individual files) | ❌ | ❌ | ✅ |
| Custom marker dictionary support | ❌ | ❌ | ✅ |

---

## 🧰 Requirements

Python 3.10+  
All dependencies are listed in `pyproject.toml`:

```bash
numpy==2.2.6
opencv-contrib-python==4.12.0.88
imutils==0.5.4
Pillow==11.3.0
reportlab==4.4.4
```

---

## 📦 Installation

### Using Poetry

```bash
poetry install
```

---

## 🎯 Usage

### 1️⃣ Generate ArUco Markers

The [`generate_aruco_markers.py`](generate_aruco_markers.py) script generates printable ArUco markers in PDF or JPG format.

#### Generate 4 default markers (IDs 0-3) as PDF
```bash
python generate_aruco_markers.py
```
Outputs: `markers/aruco_markers_DICT_5X5_100_4.pdf`

#### Generate specific number of markers
```bash
python generate_aruco_markers.py --count 10
```
Outputs: `markers/aruco_markers_DICT_5X5_100_10.pdf` (10 markers arranged in 4 columns)

#### Generate range of marker IDs
```bash
python generate_aruco_markers.py --id-range 0 15
```
Outputs: `markers/aruco_markers_DICT_5X5_100_16.pdf` (16 markers, IDs 0-15)

#### Generate specific marker IDs
```bash
python generate_aruco_markers.py --id 0 5 10 15 20
```
Outputs: `markers/aruco_markers_DICT_5X5_100_5.pdf` (5 specific markers)

#### Generate as JPG files instead of PDF
```bash
python generate_aruco_markers.py --count 4 --format jpg --output sources/
```
Outputs: `sources/marker_0000.jpg`, `sources/marker_0001.jpg`, `sources/marker_0002.jpg`, `sources/marker_0003.jpg`

#### Generate with different dictionary (6x6, 250 markers)
```bash
python generate_aruco_markers.py --dict DICT_6X6_250 --count 20
```

#### Customize layout: 5 markers per row, 300px size, 100px spacing
```bash
python generate_aruco_markers.py --count 20 --cols 5 --size 300 --spacing 100
```

##### Marker Generator Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dict` | str | `DICT_5X5_100` | ArUco dictionary (DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000, etc.) |
| `--count` | int | None | Generate N markers starting from ID 0 |
| `--id` | int[] | None | Specific marker IDs to generate (space-separated) |
| `--id-range` | int int | None | Range of marker IDs (inclusive): START END |
| `--output` | str | `markers/` | Output directory for generated markers |
| `--format` | str | `pdf` | Output format: `pdf` or `jpg` |
| `--size` | int | 200 | Marker size in pixels |
| `--spacing` | int | 50 | Spacing between markers in PDF (pixels) |
| `--cols` | int | 4 | Number of markers per row in PDF |

Available ArUco dictionaries:
- `DICT_4X4_50` - 4×4 pixels, 50 markers
- `DICT_5X5_100` - 5×5 pixels, 100 markers (recommended)
- `DICT_6X6_250` - 6×6 pixels, 250 markers
- `DICT_7X7_1000` - 7×7 pixels, 1000 markers
- `DICT_ARUCO_ORIGINAL` - Original ArUco dictionary

---

### 2️⃣ Basic AR Demo (Image-Based)

Detect 4 ArUco markers on a printed card and warp a source image onto it.

```bash
python opencv_ar_image.py --image examples/input_01.jpg --source sources/jp.jpg --output output.jpg
```

**Arguments:**
- `--image` - Input image containing the markers
- `--source` - Source image to overlay
- `--output` - Output file path

---

### 3️⃣ Enhanced AR Demo (Real-Time Webcam)

Real-time AR overlay with stabilization, FPS counter, and performance metrics.

```bash
python opencv_ar_live.py --source sources/jp.jpg
```

**Arguments:**
- `--source` - Source image to overlay (required)
- `--width` - Camera width in pixels (default: 1280)
- `--height` - Camera height in pixels (default: 720)
- `--fps` - Target FPS (default: 30)

**Controls:**
- Press `q` to quit

---

## 📂 Project Structure

```
pyar/
├── generate_aruco_markers.py   # ArUco marker generation
├── opencv_ar_image.py           # Image-based AR demo
├── opencv_ar_live.py            # Real-time webcam AR demo
├── pyproject.toml               # Poetry dependencies
├── requirements.txt             # pip dependencies
├── markers/                     # Generated marker PDFs
│   ├── aruco_markers_DICT_5X5_100_4.pdf
│   ├── aruco_markers_DICT_5X5_100_8.pdf
│   └── aruco_markers_DICT_5X5_100_16.pdf
├── sources/                     # Source images and JPG markers
│   ├── jp.jpg
│   ├── marker_0000.jpg
│   ├── marker_0001.jpg
│   └── ...
└── README.md
```

---

## 🔧 Workflow

### For End-Users:

1. **Generate markers** using [`generate_aruco_markers.py`](generate_aruco_markers.py):
   ```bash
   python generate_aruco_markers.py --count 4
   ```

2. **Print the PDF** from `markers/aruco_markers_DICT_5X5_100_4.pdf`

3. **Arrange the 4 markers** in a rectangle on a flat surface

4. **Run the live demo**:
   ```bash
   python opencv_ar_live.py --source sources/jp.jpg
   ```

5. **Position markers** in front of your webcam - the source image will warp to fit the marker area

---

## 🛠️ Technical Details

### ArUco Marker Generation

Based on: https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/

- Generates markers using OpenCV's ArUco module
- Supports multiple dictionary formats with different marker densities
- PDF output uses PIL and ReportLab for high-quality, print-ready files (300 DPI)
- JPG output saves individual marker images for flexible use
- Automatically detects and handles marker sizing and spacing

### Perspective Warping

The overlay uses homography transformation to map the source image onto the detected marker quadrilateral:

1. Detects 4 ArUco markers in the camera frame
2. Computes the center of each marker
3. Orders the centers to form the quadrilateral (top-left, top-right, bottom-right, bottom-left)
4. Calculates homography matrix between source corners and detected quadrilateral
5. Warps source image using perspective transformation
6. Blends the warped image with the camera frame using a mask

### Stabilization (Enhanced Mode)

Exponential smoothing is applied to the homography matrix to reduce jitter:
```
H_smooth = α * H_current + (1 - α) * H_previous
```

Where `α` is typically 0.1-0.3 for smooth, stable overlays.

---

## 📋 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.2.6 | Numerical computations, matrix operations |
| opencv-contrib-python | 4.12.0.88 | Computer vision, ArUco markers |
| imutils | 0.5.4 | Image processing utilities |
| Pillow | 11.3.0 | Image handling (PNG, JPG, PDF) |
| reportlab | 4.4.4 | PDF generation and rendering |

---

## 🐛 Troubleshooting

**Markers not detected?**
- Ensure markers are clearly visible and well-lit
- Check that all 4 markers are in the camera frame
- Try adjusting lighting or marker size

**Warped image distorted?**
- Markers must form a roughly rectangular shape
- Markers should be on a flat surface
- Camera lens distortion may require calibration

**PDF generation fails?**
- Check write permissions in output directory
- Ensure enough disk space available
- Try using JPG format instead: `--format jpg`

**Poor performance in live mode?**
- Reduce camera resolution: `--width 640 --height 480`
- Lower target FPS: `--fps 15`
- Ensure adequate lighting for marker detection

---

## 📄 License

MIT

---

## 🙏 References

- **ArUco Markers Guide**: https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/
- **OpenCV ArUco Module**: https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
- **Perspective Transformation**: https://docs.opencv.org/master/d9/df8/tutorial_root.html

