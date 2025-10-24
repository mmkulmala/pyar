#!/usr/bin/env python3
"""
opencv_ar_live_stable.py
Real-time Augmented Reality demo using OpenCV and ArUco markers,
with exponential smoothing (stabilization) for steady overlay.

Usage:
    python opencv_ar_live_stable.py --source sources/marker_0000.jpg
Press 'q' to quit.
"""

import cv2
import numpy as np
import argparse
import sys

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def overlay_warped(source, frame, dst_quad):
    (h_src, w_src) = source.shape[:2]
    src_pts = np.array([[0,0],[w_src-1,0],[w_src-1,h_src-1],[0,h_src-1]], dtype="float32")
    dst_pts = order_points(dst_quad)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return frame

    warped = cv2.warpPerspective(source, H, (frame.shape[1], frame.shape[0]))
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(int), 255)

    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(warped, warped, mask=mask)
    return cv2.add(bg, fg)

def find_quad_from_aruco(corners):
    if len(corners) < 4:
        return None
    centers = [c[0].mean(axis=0) for c in corners]
    centers = np.array(centers)
    rect = order_points(centers)
    return rect

def smooth_quad(prev_quad, new_quad, alpha=0.7):
    """Exponential smoothing between previous and new quad"""
    if prev_quad is None:
        return new_quad
    return alpha * prev_quad + (1 - alpha) * new_quad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True, help="path to source image to overlay")
    ap.add_argument("-c", "--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--smooth", type=float, default=0.7,
                    help="smoothing factor (0=no smoothing, 0.7=strong default)")
    args = vars(ap.parse_args())

    source = cv2.imread(args["source"])
    if source is None:
        print("[ERROR] Could not load source image:", args["source"])
        sys.exit(1)

    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(args["camera"])
    if not cap.isOpened():
        print("[ERROR] Cannot access camera")
        sys.exit(1)

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()

    prev_quad = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed, skipping...")
            continue

        (corners, ids, _) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        if ids is not None and len(corners) >= 4:
            quad = find_quad_from_aruco(corners[:4])
            if quad is not None:
                # Apply smoothing
                quad = smooth_quad(prev_quad, quad, alpha=args["smooth"])
                prev_quad = quad
                frame = overlay_warped(source, frame, quad)
        else:
            # Reset if markers lost
            prev_quad = None

        # Draw markers for feedback
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow("AR Live (Stabilized)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] Quitting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
