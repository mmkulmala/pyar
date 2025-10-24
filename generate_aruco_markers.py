#!/usr/bin/env python3
"""
generate_aruco_markers.py
Generate ArUco markers and save them as PDF and/or JPG files.

Based on: https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/

Usage:
    python generate_aruco_markers.py --dict DICT_5X5_100 --count 10 --output sources/
    python generate_aruco_markers.py --dict DICT_5X5_100 --id-range 0 15 --format jpg
    python generate_aruco_markers.py --dict DICT_5X5_100 --id 0 5 10 15 --format both
    python generate_aruco_markers.py --dict DICT_5X5_100 --id-range 0 99 --cols 5 --output sources/
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def check_aruco_available():
    """Check if cv2.aruco module is available."""
    try:
        # Test if aruco module exists and has getPredefinedDictionary
        if not hasattr(cv2, 'aruco'):
            return False, "cv2.aruco module not found"
        if not hasattr(cv2.aruco, 'getPredefinedDictionary'):
            return False, "cv2.aruco.getPredefinedDictionary not available"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def generate_single_marker(marker_id, aruco_dict, marker_size=200):
    """
    Generate a single ArUco marker image.
    
    Args:
        marker_id: Integer ID of the marker
        aruco_dict: cv2.aruco dictionary object
        marker_size: Size in pixels for the marker
        
    Returns:
        numpy array containing the marker image
    """
    # Prefer cv2.aruco.drawMarker when available (newer OpenCV versions),
    # otherwise fall back to generateImageMarker. As a last resort attempt
    # to read from the dictionary bytesList.
    try:
        # drawMarker (returns an image) is available in newer OpenCV
        if hasattr(cv2.aruco, 'drawMarker'):
            try:
                marker_image = cv2.aruco.drawMarker(aruco_dict, int(marker_id), int(marker_size))
                # ensure grayscale uint8
                marker_image = np.asarray(marker_image)
                if marker_image.dtype != np.uint8:
                    marker_image = (marker_image * 255).astype(np.uint8)
                if marker_image.ndim == 3 and marker_image.shape[2] == 3:
                    marker_image = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)
                return marker_image
            except Exception:
                pass

        # Older versions may provide generateImageMarker
        if hasattr(cv2.aruco, 'generateImageMarker'):
            try:
                marker_image = cv2.aruco.generateImageMarker(aruco_dict, int(marker_id), int(marker_size))
                return marker_image
            except Exception:
                pass

        # Try constructing from bytesList (low-level); this may work depending on the cv2 build
        try:
            bits = aruco_dict.bytesList[int(marker_id)]
            arr = np.frombuffer(bits, dtype=np.uint8)
            # unpack bits to bitmap
            bits_unpacked = np.unpackbits(arr)
            side = int(np.sqrt(bits_unpacked.size))
            bitmap = bits_unpacked.reshape((side, side)).astype(np.uint8) * 255
            marker_image = cv2.resize(bitmap, (int(marker_size), int(marker_size)), interpolation=cv2.INTER_NEAREST)
            return marker_image
        except Exception:
            pass

        raise RuntimeError("No available ArUco marker generator function in this OpenCV build.")
    except Exception as e:
        raise RuntimeError(f"Failed to generate marker: {e}")


def save_individual_jpg(marker_image, marker_id, output_path, quality=95):
    """
    Save individual marker as JPG file.
    
    Args:
        marker_image: numpy array with marker image
        marker_id: Integer ID of the marker
        output_path: Path to output directory
        quality: JPG quality (1-100)
    """
    jpg_path = output_path / f"marker_{marker_id:04d}.jpg"
    cv2.imwrite(str(jpg_path), marker_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpg_path


def save_individual_png(marker_image, marker_id, output_path):
    """
    Save individual marker as PNG file.
    
    Args:
        marker_image: numpy array with marker image
        marker_id: Integer ID of the marker
        output_path: Path to output directory
    """
    png_path = output_path / f"marker_{marker_id:04d}.png"
    cv2.imwrite(str(png_path), marker_image)
    return png_path


def create_pdf_with_markers(markers_dict, output_path, marker_size=200, spacing=50, markers_per_row=4):
    """
    Create a PDF containing all generated markers arranged in a grid.
    
    Args:
        markers_dict: Dictionary with marker_id as key and marker image as value
        output_path: Path to save the PDF
        marker_size: Size of each marker in pixels
        spacing: Space between markers in pixels
        markers_per_row: Number of markers to display per row
    """
    from PIL import Image
    
    # Calculate grid dimensions
    num_markers = len(markers_dict)
    num_rows = (num_markers + markers_per_row - 1) // markers_per_row
    
    # Calculate dimensions at high DPI for print quality
    dpi = 300
    pixel_to_point = 72.0 / dpi
    scale = dpi / 96
    
    scaled_marker_size = int(marker_size * scale)
    scaled_spacing = int(spacing * scale)
    label_height = int(30 * scale)
    
    # Calculate page dimensions
    content_width = markers_per_row * scaled_marker_size + (markers_per_row + 1) * scaled_spacing
    content_height = num_rows * (scaled_marker_size + label_height) + (num_rows + 1) * scaled_spacing
    
    page_width = int(content_width * pixel_to_point)
    page_height = int(content_height * pixel_to_point)
    
    # Create PIL images for each marker at high DPI
    pil_images = []
    for marker_id in sorted(markers_dict.keys()):
        marker_cv = markers_dict[marker_id]
        
        # Resize marker to high DPI
        marker_resized = cv2.resize(marker_cv, (scaled_marker_size, scaled_marker_size))
        
        # Convert from grayscale to RGB for PIL
        marker_rgb = cv2.cvtColor(marker_resized, cv2.COLOR_GRAY2RGB)
        marker_pil = Image.fromarray(marker_rgb)
        pil_images.append((marker_id, marker_pil))
    
    # Create a blank white page
    page = Image.new('RGB', (page_width, page_height), color='white')
    draw = ImageDraw.Draw(page)
    
    # Paste markers onto page
    x_offset = scaled_spacing
    y_offset = scaled_spacing
    
    for idx, (marker_id, marker_pil) in enumerate(pil_images):
        row = idx // markers_per_row
        col = idx % markers_per_row
        
        x = int((x_offset + col * (scaled_marker_size + scaled_spacing)) * pixel_to_point)
        y = int((y_offset + row * (scaled_marker_size + scaled_spacing + label_height)) * pixel_to_point)
        
        # Paste marker
        page.paste(marker_pil, (x, y))
        
        # Add text label below marker
        label_x = x + scaled_marker_size // 2
        label_y = y + scaled_marker_size + int(5 * scale)
        label_text = f"ID: {marker_id}"
        
        try:
            draw.text((label_x, label_y), label_text, fill='black', anchor='lm')
        except:
            pass  # Font rendering might fail on some systems
    
    # Save as PDF at high DPI
    page.save(str(output_path), dpi=(dpi, dpi))
    print(f"✓ PDF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple ArUco markers and save as PDF and/or JPG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate first 10 markers as PDF to sources folder
  python generate_aruco_markers.py --count 10 --output sources/

  # Generate specific range of marker IDs as JPG files
  python generate_aruco_markers.py --id-range 0 15 --format jpg --output sources/

  # Generate 100 markers as both PDF and individual JPG files
  python generate_aruco_markers.py --count 100 --cols 5 --format both --output sources/

  # Generate with different dictionary and PNG format
  python generate_aruco_markers.py --dict DICT_6X6_250 --count 20 --format jpg
        """
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_5X5_100",
        help="ArUco dictionary to use (default: DICT_5X5_100)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Generate N markers starting from ID 0"
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs='+',
        dest="marker_ids",
        default=None,
        help="Specific marker IDs to generate (space-separated)"
    )
    parser.add_argument(
        "--id-range",
        type=int,
        nargs=2,
        dest="id_range",
        metavar=("START", "END"),
        default=None,
        help="Range of marker IDs to generate (inclusive)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sources/",
        help="Output directory for markers (default: sources/)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['pdf', 'jpg', 'png', 'both'],
        default='jpg',
        help="Output format: pdf, jpg, png, or both (default: jpg)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Size of each marker in pixels (default: 200)"
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=50,
        help="Spacing between markers in PDF (default: 50)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of markers per row in PDF (default: 4)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPG quality 1-100 (default: 95)"
    )
    
    args = parser.parse_args()
    
    # Check if cv2.aruco is available
    aruco_available, check_msg = check_aruco_available()
    if not aruco_available:
        print(f"[ERROR] cv2.aruco not available. Make sure opencv-contrib-python is installed")
        print(f"Details: {check_msg}")
        print(f"\nYour OpenCV version: {cv2.__version__}")
        print(f"To fix this, run:")
        print(f"  pip install --upgrade opencv-contrib-python")
        print(f"  or use: bash install.sh")
        return 1
    
    # Get ArUco dictionary
    aruco_dict_name = args.dict
    try:
        if not hasattr(cv2.aruco, aruco_dict_name):
            print(f"✗ Error: Unknown ArUco dictionary '{aruco_dict_name}'")
            print("Available dictionaries:")
            print("  DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL")
            return 1
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    except Exception as e:
        print(f"✗ Error loading ArUco dictionary: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"✗ Error creating output directory: {e}")
        return 1
    
    # Determine which markers to generate
    marker_ids = None
    if args.marker_ids is not None:
        marker_ids = sorted(args.marker_ids)
    elif args.id_range is not None:
        start, end = args.id_range
        marker_ids = list(range(start, end + 1))
    elif args.count is not None:
        marker_ids = list(range(args.count))
    else:
        # Default: generate first 4 markers
        marker_ids = [0, 1, 2, 3]
    
    print(f"Using ArUco dictionary: {aruco_dict_name}")
    print(f"Generating {len(marker_ids)} markers with IDs: {marker_ids}")
    print(f"Output format: {args.format}")
    
    # Generate markers
    markers_dict = {}
    jpg_count = 0
    png_count = 0
    
    for marker_id in marker_ids:
        try:
            marker_image = generate_single_marker(marker_id, aruco_dict, args.size)
            markers_dict[marker_id] = marker_image
            print(f"✓ Generated marker ID {marker_id}")
            
            # Save as JPG if requested
            if args.format in ['jpg', 'both']:
                jpg_path = save_individual_jpg(marker_image, marker_id, output_dir, args.quality)
                jpg_count += 1
            
            # Save as PNG if requested
            if args.format in ['png', 'both']:
                png_path = save_individual_png(marker_image, marker_id, output_dir)
                png_count += 1
                
        except Exception as e:
            print(f"✗ Error generating marker {marker_id}: {e}")
            continue
    
    if not markers_dict:
        print("✗ No markers were generated successfully.")
        return 1
    
    print(f"\n✓ Generated {len(markers_dict)} markers successfully.")
    
    # Create PDF with all markers if requested
    if args.format in ['pdf', 'both']:
        pdf_output_path = output_dir / f"aruco_markers_{aruco_dict_name}_{len(markers_dict)}.pdf"
        try:
            print(f"Creating PDF with {args.cols} markers per row...")
            create_pdf_with_markers(
                markers_dict,
                str(pdf_output_path),
                marker_size=args.size,
                spacing=args.spacing,
                markers_per_row=args.cols
            )
        except Exception as e:
            print(f"✗ Error creating PDF: {e}")
            return 1
    
    # Print summary
    print(f"\n✓ Marker generation complete!")
    if jpg_count > 0:
        print(f"✓ Saved {jpg_count} JPG files")
    if png_count > 0:
        print(f"✓ Saved {png_count} PNG files")
    if args.format in ['pdf', 'both']:
        print(f"✓ Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
