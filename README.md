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

- Python 3.10+
- pip (Python package manager)

All dependencies are listed in [`requirements.txt`](requirements.txt):

```
numpy==2.2.6
opencv-contrib-python==4.12.0.88
imutils==0.5.4
Pillow==11.3.0
reportlab==4.4.4
```

---

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
git clone <repository-url>
cd pyar
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages: numpy, opencv-contrib-python, imutils, Pillow, and reportlab.

### Verify Installation

```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

---

## 🎯 Usage

### 1️⃣ Generate ArUco Markers

The [`generate_aruco_markers.py`](generate_aruco_markers.py) script generates printable ArUco markers in PDF or JPG format.

#### Generate 4 default markers (IDs 0-3) as PDF
```bash
python generate_aruco_markers.py
```
**Output:** `markers/aruco_markers_DICT_5X5_100_4.pdf`

#### Generate specific number of markers
```bash
python generate_aruco_markers.py --count 10
```
**Output:** `markers/aruco_markers_DICT_5X5_100_10.pdf` (10 markers arranged in 4 columns)

#### Generate range of marker IDs
```bash
python generate_aruco_markers.py --id-range 0 15
```
**Output:** `markers/aruco_markers_DICT_5X5_100_16.pdf` (16 markers, IDs 0-15)

#### Generate specific marker IDs
```bash
python generate_aruco_markers.py --id 0 5 10 15 20
```
**Output:** `markers/aruco_markers_DICT_5X5_100_5.pdf` (5 specific markers)

#### Generate as JPG files instead of PDF
```bash
python generate_aruco_markers.py --count 4 --format jpg --output sources/
```
**Output:** `sources/marker_0000.jpg`, `sources/marker_0001.jpg`, `sources/marker_0002.jpg`, `sources/marker_0003.jpg`

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

**Available ArUco Dictionaries:**
- `DICT_4X4_50` - 4×4 pixels, 50 markers
- `DICT_5X5_100` - 5×5 pixels, 100 markers (recommended)
- `DICT_6X6_250` - 6×6 pixels, 250 markers
- `DICT_7X7_1000` - 7×7 pixels, 1000 markers
- `DICT_ARUCO_ORIGINAL` - Original ArUco dictionary

---

### 2️⃣ Image-Based AR Demo (Static Image Processing)

Detect 4 ArUco markers on a printed card and warp a source image onto it. This script processes static images without requiring a webcam.

#### Step 1: Prepare Your Input Image

Your input image should contain a printed card with 4 ArUco markers arranged in a rectangle:

```
┌─────────────────────────────────┐
│  ID:0          ID:1            │
│   (top-left)   (top-right)     │
│                                  │
│                                  │
│  ID:3          ID:2            │
│ (bottom-left) (bottom-right)   │
└─────────────────────────────────┘
```

The script uses the **center of each marker** to define card corners, not the marker edges themselves. This provides more accurate perspective warping.

#### Step 2: Run Basic Overlay (Single Source Image)

```bash
python opencv_ar_image.py --image test_marker_card.jpg --source sources/marker_0000.jpg --output output/result.jpg
```

This creates:
- `output/result_marker_0000.jpg` - AR-overlaid image with source warped onto detected card

#### Step 3: Add Debug Visualization

To see which markers were detected and verify the perspective quad:

```bash
python opencv_ar_image.py --image test_marker_card.jpg --source sources/marker_0000.jpg --output output/result.jpg --debug
```

This additionally creates:
- `output/result_annotated.jpg` - Debug visualization showing detected markers, marker IDs, and quad outline

#### Step 4: Batch Process Multiple Source Images

If you have multiple source images to overlay:

```bash
python opencv_ar_image.py --image test_marker_card.jpg --source sources/ --output output/batch
```

Generates:
- `output/batch_marker_0000.jpg`
- `output/batch_marker_0001.jpg`
- `output/batch_marker_0002.jpg`
- etc. (one for each image in sources/)

#### Step 5: Troubleshoot with Verbose Debug

If detection fails or produces unexpected results:

```bash
python opencv_ar_image.py --image problematic_card.jpg --source image.jpg --output output/debug --debug --debug-verbose
```

The `--debug-verbose` flag prints per-dictionary detection counts to help identify detection issues.

#### Step 6: Force Specific Dictionary (If Auto-Detection Fails)

```bash
python opencv_ar_image.py --image card.jpg --source image.jpg --output output/result.jpg --dict DICT_6X6_250
```

Supported dictionaries: `DICT_4X4_50`, `DICT_5X5_100`, `DICT_6X6_250`, `DICT_7X7_1000`, `DICT_ARUCO_ORIGINAL`

#### opencv_ar_image.py Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` / `-i` | str | **required** | Input image containing the ArUco markers (printed card) |
| `--source` / `-s` | str or dir | **required** | Source image (or directory of images) to overlay on the card |
| `--output` / `-o` | str | `output.jpg` | Output file path or base name for batch processing |
| `--dict` | str | auto-detect | Force a specific ArUco dictionary (e.g., `DICT_5X5_100`, `DICT_6X6_250`) |
| `--debug` | flag | disabled | Save annotated debug images showing detected markers and quad overlay |
| `--debug-verbose` | flag | disabled | Print per-dictionary detection counts during auto-selection |
| `--no-show` | flag | disabled | Don't display the result image in a GUI window |

#### How It Works

1. **Loads** the input image containing printed ArUco markers
2. **Detects** all 4 ArUco markers (IDs 0-3) using the optimal dictionary
3. **Computes** homography transformation from source image corners to detected marker centers
4. **Warps** the source image using perspective transformation
5. **Blends** the warped image onto the original using a mask
6. **Saves** the result (and optional debug annotation)

#### Common Use Cases

**Single image overlay:**
```bash
python opencv_ar_image.py --image card_photo.jpg --source logo.png --output output/card_with_logo.jpg
```

**Batch watermarking (multiple source images):**
```bash
python opencv_ar_image.py --image template_card.jpg --source watermarks/ --output output/watermarked
```

**Troubleshooting with debug output:**
```bash
python opencv_ar_image.py --image problematic_card.jpg --source image.jpg --output output/debug --debug --debug-verbose
```

---

### 3️⃣ Enhanced AR Demo (Real-Time Webcam)

Real-time AR overlay with stabilization, FPS counter, and performance metrics.

```bash
python opencv_ar_live.py --source sources/marker_0000.jpg
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
├── opencv_ar_image.py          # Image-based AR demo
├── opencv_ar_live.py           # Real-time webcam AR demo
├── requirements.txt            # pip dependencies
├── README.md                   # This file
├── markers/                    # Generated marker PDFs
│   ├── aruco_markers_DICT_5X5_100_4.pdf
│   ├── aruco_markers_DICT_5X5_100_8.pdf
│   └── aruco_markers_DICT_5X5_100_16.pdf
├── output/                     # Generated output images
│   ├── example_output_marker_0000.jpg
│   └── example_output_annotated.jpg
└── sources/                    # Source images and JPG markers
    ├── marker_0000.jpg
    ├── marker_0001.jpg
    └── ...
```

---

## 🔧 Complete Example Workflow

### Step-by-Step Guide: Generate Markers → Print → Run AR Demo

#### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

#### 2. **Generate Markers (4 default markers)**

```bash
python generate_aruco_markers.py
```

This creates `markers/aruco_markers_DICT_5X5_100_4.pdf` with 4 printable markers.

#### 3. **Print the PDF**

- Open `markers/aruco_markers_DICT_5X5_100_4.pdf`
- Print at 100% scale (important for marker recognition)
- Use a white background for best results

#### 4. **Arrange Markers**

Arrange the 4 printed markers in a roughly rectangular pattern on a flat surface:

```
┌─────────────────────────────────┐
│  ID:0          ID:1             │
│                                 │
│                                 │
│  ID:3          ID:2             │
└─────────────────────────────────┘
```

Ensure:
- All 4 markers are visible to your webcam
- Markers form a clear rectangular shape
- Good lighting (avoid shadows and glare)

#### 5. **Run the Live AR Demo**

```bash
python opencv_ar_live.py --source sources/marker_0000.jpg
```

#### 6. **Position Your Webcam**

- Place your camera to see all 4 markers
- The source image will warp to fit inside the marker rectangle
- Press `q` to quit

---

## 📊 Workflow Examples

### Generate Multiple Marker Sets for Different Use Cases

```bash
# Set 1: 4 markers for testing (default)
python generate_aruco_markers.py

# Set 2: 16 markers for documentation (4x4 grid)
python generate_aruco_markers.py --count 16 --cols 4

# Set 3: Individual JPG markers for flexible arrangement
python generate_aruco_markers.py --count 4 --format jpg --output sources/

# Set 4: Large 6x6 dictionary with 20 markers
python generate_aruco_markers.py --dict DICT_6X6_250 --count 20 --size 300
```

### Then Use with AR Demo

```bash
# Run with default markers
python opencv_ar_live.py --source sources/marker_0000.jpg

# With custom resolution (for older cameras)
python opencv_ar_live.py --source sources/marker_0000.jpg --width 640 --height 480 --fps 20
```

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

**Installation fails?**
- Check Python version: `python --version` (requires 3.10+)
- Ensure pip is up to date: `pip install --upgrade pip`
- Try installing dependencies individually: `pip install numpy opencv-contrib-python imutils Pillow reportlab`
- Check write permissions in project directory

**Markers not detected?**
- Ensure markers are clearly visible and well-lit
- Check that all 4 markers are in the camera frame
- Try adjusting lighting or marker size
- Print at 100% scale (not scaled down)
- Use `--debug-verbose` to identify which dictionary detects markers

**Warped image distorted?**
- Markers must form a roughly rectangular shape
- Markers should be on a flat surface
- Camera lens distortion may require calibration
- Try adjusting marker positions

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
