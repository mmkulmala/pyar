#!/usr/bin/env python3
"""
opencv_ar_image.py
Image-based AR demo: detect 4 ArUco markers on a printed card and warp a source image onto it.
Usage:
    poetry run python opencv_ar_image.py --image examples/input_01.jpg --source sources/jp.jpg --output output.jpg
"""

import argparse
import sys
import cv2
import numpy as np
import imutils

def order_points(pts):
    # pts: list/array shape (4,2) unordered.
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def get_destination_coords(w, h):
    # destination points in order: tl, tr, br, bl
    return np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

def find_quad_from_aruco(corners, ids):
    # corners is list of 4 arrays (one per detected marker), ids is Nx1
    # we expect exactly 4 markers; return sorted (tl,tr,br,bl) coordinates of the card corners
    # each marker corner array shape: (1,4,2) -> use its center for ordering markers
    if len(corners) != 4:
        return None

    # compute marker centers
    centers = []
    for c in corners:
        c = c.reshape((4, 2))
        center = c.mean(axis=0)
        centers.append(center)

    # Convert to array then sort to get approximate card orientation
    centers = np.array(centers)
    # Use the four centers to compute the convex hull corners (approx)
    hull = cv2.convexHull(centers.astype(np.float32))
    if hull.shape[0] < 4:
        # fallback: use centers sorted by sum/diff (top-left etc)
        rect = order_points(centers)
        return rect

    # hull may have more points if collinear; extract extremes by projecting
    # Use ordering by sum/diff to be robust
    rect = order_points(centers)
    return rect

def overlay_warped(source, target_image, dst_quad):
    (h_src, w_src) = source.shape[:2]
    dst_pts = order_points(dst_quad)
    src_pts = get_destination_coords(w_src, h_src)

    # compute homography from source to destination quad
    H, status = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        raise RuntimeError("Could not compute homography")

    # warp source into target image space
    warped = cv2.warpPerspective(source, H, (target_image.shape[1], target_image.shape[0]))

    # create mask from warped image (non-black pixels)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(target_image, target_image, mask=mask_inv)
    img_fg = cv2.bitwise_and(warped, warped, mask=mask)

    out = cv2.add(img_bg, img_fg)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUco tag (card)")
    ap.add_argument("-s", "--source", required=True, help="path to input source image to place on the card")
    ap.add_argument("-o", "--output", required=False, default="output.jpg", help="path to write augmented output")
    args = vars(ap.parse_args())

    print("[INFO] loading images...")
    image = cv2.imread(args["image"])
    if image is None:
        print("[ERROR] could not load input image:", args["image"])
        sys.exit(1)
    image = imutils.resize(image, width=800)

    source = cv2.imread(args["source"])
    if source is None:
        print("[ERROR] could not load source image:", args["source"])
        sys.exit(1)

    print("[INFO] detecting ArUco markers...")
    # choose dictionary used for the printed markers; the tutorial uses DICT_ARUCO_ORIGINAL
    try:
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    except AttributeError:
        print("[ERROR] cv2.aruco not available. Make sure opencv-contrib-python is installed.")
        sys.exit(1)

    if ids is None or len(corners) < 4:
        print("[INFO] could not find 4 markers (found {}). Exiting.".format(0 if ids is None else len(corners)))
        sys.exit(0)

    # If more than 4 markers found, we need to pick the four that form the card.
    # Simplest heuristic: take the 4 markers with the largest convex hull area of centers, or pick first four.
    if len(corners) > 4:
        # compute centers and choose 4 that form largest spread (convex hull area)
        centers = [c.reshape((4,2)).mean(axis=0) for c in corners]
        centers_arr = np.array(centers)
        # compute pairwise distances sums
        sums = centers_arr.sum(axis=1)
        # choose 4 with median sums around center? simpler: choose 4 with max variance
        variances = np.var(centers_arr, axis=1)
        idx = np.argsort(-variances)[:4]
        chosen_corners = [corners[i] for i in idx]
    else:
        chosen_corners = corners

    quad = find_quad_from_aruco(chosen_corners, ids)
    if quad is None:
        print("[ERROR] Could not compute card quadrilateral.")
        sys.exit(1)

    print("[INFO] warping source onto detected card...")
    try:
        out = overlay_warped(source, image, quad)
    except Exception as e:
        print("[ERROR] warping failed:", e)
        sys.exit(1)

    print("[INFO] saving result to", args["output"])
    cv2.imwrite(args["output"], out)
    # show result
    cv2.imshow("Augmented", out)
    print("[INFO] press any key in image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
