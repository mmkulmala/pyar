#!/usr/bin/env python3
"""
opencv_ar_image.py
Image-based AR demo: detect 4 ArUco markers on a printed card and warp a source image onto it.
Usage:
    python opencv_ar_image.py --image examples/input1.jpg --source sources/marker_0000.jpg --output output.jpg
"""

import argparse
import sys
import os
import glob
import cv2
import numpy as np

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # resize image while keeping aspect ratio (replacement for imutils.resize)
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

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

    # combine background and warped foreground
    out = cv2.add(img_bg, img_fg)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUco tag (card)")
    ap.add_argument("-s", "--source", required=True, help="path to input source image to place on the card")
    ap.add_argument("--dict", required=False, default=None, help="(optional) ArUco dictionary name to force, e.g. DICT_5X5_100")
    ap.add_argument("-o", "--output", required=False, default="output.jpg", help="path to write augmented output")
    ap.add_argument("--no-show", dest="no_show", action="store_true", help="don't open a GUI window to show the result")
    ap.add_argument("--debug", dest="debug", action="store_true", help="save an annotated debug image showing detected markers and ids")
    # verbose debug: print per-dictionary detection counts while trying candidates
    ap.add_argument("--debug-verbose", "--debug_verbose", dest="debug_verbose", action="store_true", help="print per-dictionary detection counts during dictionary auto-selection")
    args = vars(ap.parse_args())

    print("[INFO] loading images...")
    image = cv2.imread(args["image"])
    if image is None:
        print("[ERROR] could not load input image:", args["image"])
        sys.exit(1)
    image = resize_image(image, width=800)

    # Note: --source can be a single image path or a directory containing multiple source images.
    # We'll detect the card once from the input image and then iterate all sources.

    print("[INFO] detecting ArUco markers...")
    # robustly access the aruco module and handle API differences across OpenCV versions
    if not hasattr(cv2, 'aruco'):
        print("[ERROR] cv2.aruco not available. This OpenCV build doesn't include aruco (contrib) module.")
        print("[ERROR] Install the contrib build: pip install --upgrade opencv-contrib-python")
        sys.exit(1)

    aruco = cv2.aruco

    # create detector parameters (API differences across versions)
    if hasattr(aruco, 'DetectorParameters_create'):
        arucoParams = aruco.DetectorParameters_create()
    elif hasattr(aruco, 'DetectorParameters'):
        # some builds expose the class/constructor directly
        try:
            arucoParams = aruco.DetectorParameters()
        except Exception:
            arucoParams = None
    else:
        arucoParams = None

    # tune a couple of settings if available
    if arucoParams is not None:
        if hasattr(arucoParams, 'cornerRefinementMethod') and hasattr(aruco, 'CORNER_REFINE_SUBPIX'):
            try:
                arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            except Exception:
                pass
        try:
            # adaptiveThreshConstant exists on many versions
            arucoParams.adaptiveThreshConstant = 7
        except Exception:
            pass

    # obtain a dictionary in a version-tolerant way
    # If user passed a specific dictionary name, try that first. Otherwise try a list of common dictionaries
    dict_override = args.get("dict")
    candidate_names = []
    if dict_override:
        candidate_names = [dict_override]
    else:
        # common dictionaries to try (ordered by likelihood for printed card sets in this repo)
        candidate_names = [
            'DICT_5X5_100', 'DICT_5X5_50', 'DICT_4X4_50', 'DICT_4X4_100',
            'DICT_6X6_250', 'DICT_7X7_100', 'DICT_ARUCO_ORIGINAL'
        ]

    # helper to create dictionary object from name if available
    def make_dict(name):
        if not hasattr(aruco, name):
            return None
        dict_id = getattr(aruco, name)
        if hasattr(aruco, 'getPredefinedDictionary'):
            try:
                return aruco.getPredefinedDictionary(dict_id)
            except Exception:
                pass
        if hasattr(aruco, 'Dictionary_get'):
            try:
                return aruco.Dictionary_get(dict_id)
            except Exception:
                pass
        return None

    # convert to grayscale for detection (detectMarkers expects single-channel images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    best = None
    best_name = None
    best_detect = (None, None, None)
    dict_counts = []
    # Try each candidate dictionary and pick the one that yields the most detected markers
    for name in candidate_names:
        arucoDict_try = make_dict(name)
        if arucoDict_try is None:
            continue
        try:
            (corners_try, ids_try, rejected_try) = aruco.detectMarkers(gray, arucoDict_try, parameters=arucoParams)
        except Exception:
            continue
        count = 0 if ids_try is None else len(ids_try)
        dict_counts.append((name, count))
        if best is None or count > best:
            best = count
            best_name = name
            best_detect = (corners_try, ids_try, rejected_try)

    # If verbose debug requested, print per-dictionary detection counts
    if args.get("debug_verbose"):
        print("[DEBUG-VERBOSE] detection counts per candidate dictionary:")
        for n, c in dict_counts:
            print(f" - {n}: {c}")

    if best is None or best == 0:
        print("[INFO] could not find any markers with known dictionaries. Exiting.")
        sys.exit(0)

    (corners, ids, rejected) = best_detect
    print(f"[DEBUG] cv2.aruco available, proceeding with marker detection using {best_name} (found {0 if ids is None else len(ids)})")

    # detection chosen from candidate dictionaries (see debug message above)

    if ids is None or len(corners) < 4:
        print("[INFO] could not find 4 markers (found {}). Exiting.".format(0 if ids is None else len(corners)))
        # optionally save debug annotated image showing detections (if any)
        if args.get("debug"):
            debug_path = os.path.splitext(args.get("output"))[0] + "_annotated.jpg"
            disp = image.copy()
            try:
                aruco.drawDetectedMarkers(disp, corners, ids)
                cv2.imwrite(debug_path, disp)
                print(f"[DEBUG] annotated image saved to {debug_path}")
            except Exception:
                pass
        sys.exit(0)

    # Always use marker IDs to map corners: expect IDs 0,1,2,3 for sources/marker_0000.jpg ... marker_0003.jpg
    expected_ids = [0, 1, 2, 3]
    found_ids = [int(i[0]) for i in ids] if ids is not None else []
    if sorted(found_ids) != expected_ids:
        print(f"[ERROR] Detected marker IDs {found_ids} do not match expected {expected_ids}. Card mapping will be incorrect.")
        if args.get("debug"):
            debug_path = os.path.splitext(args.get("output"))[0] + "_annotated.jpg"
            disp = image.copy()
            try:
                aruco.drawDetectedMarkers(disp, corners, ids)
                cv2.imwrite(debug_path, disp)
                print(f"[DEBUG] annotated image saved to {debug_path}")
            except Exception:
                pass
        sys.exit(1)

    # FIX: Use marker centers to define card corners, not arbitrary marker corners
    # Build quad for the card using detected marker centers
    # Mapping: marker_0000.jpg = top-left, marker_0001.jpg = top-right, marker_0002.jpg = bottom-right, marker_0003.jpg = bottom-left
    id_to_corner = {0:0, 1:1, 2:2, 3:3}  # 0=tl, 1=tr, 2=br, 3=bl
    quad = np.zeros((4,2), dtype="float32")
    for marker_id, card_corner in id_to_corner.items():
        if marker_id not in found_ids:
            print(f"[ERROR] Expected marker ID {marker_id} not found in detected IDs {found_ids}.")
            if args.get("debug"):
                debug_path = os.path.splitext(args.get("output"))[0] + "_annotated.jpg"
                disp = image.copy()
                try:
                    aruco.drawDetectedMarkers(disp, corners, ids)
                    cv2.imwrite(debug_path, disp)
                    print(f"[DEBUG] annotated image saved to {debug_path}")
                except Exception:
                    pass
            sys.exit(1)
        marker_idx = found_ids.index(marker_id)
        c = corners[marker_idx].reshape((4,2))
        # Use the center of each marker as the card corner point
        center = c.mean(axis=0)
        quad[card_corner] = center

    if quad is None or len(quad) != 4:
        print("[ERROR] Could not compute card quadrilateral.")
        sys.exit(1)

    # optional debug: save annotated image with detected markers and chosen quad
    if args.get("debug"):
        try:
            disp = image.copy()
            aruco.drawDetectedMarkers(disp, corners, ids)
            # draw quad
            q = quad.astype(int)
            cv2.polylines(disp, [q.reshape((-1,1,2))], isClosed=True, color=(0,255,0), thickness=2)
            debug_path = os.path.splitext(args.get("output"))[0] + "_annotated.jpg"
            cv2.imwrite(debug_path, disp)
            print(f"[DEBUG] annotated image saved to {debug_path}")
        except Exception:
            pass

    # Prepare list of source images. If a directory is given, collect files inside it.
    src_arg = args["source"]
    if os.path.isdir(src_arg):
        pattern = os.path.join(src_arg, '*')
        files = sorted(glob.glob(pattern))
        # filter common image extensions
        source_paths = [f for f in files if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
        if len(source_paths) == 0:
            print("[ERROR] no image files found in source directory:", src_arg)
            sys.exit(1)
    else:
        source_paths = [src_arg]

    # Prepare output naming
    out_arg = args["output"]
    out_dir = os.path.dirname(out_arg) or '.'
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    base_name = os.path.splitext(os.path.basename(out_arg))[0]
    base_ext = os.path.splitext(out_arg)[1] or '.jpg'

    last_out = None
    for src_path in source_paths:
        source = cv2.imread(src_path)
        if source is None:
            print("[WARNING] could not load source image, skipping:", src_path)
            continue

        print(f"[INFO] warping source {os.path.basename(src_path)} onto detected card...")
        try:
            out = overlay_warped(source, image, quad)
        except Exception as e:
            print("[ERROR] warping failed for {}: {}".format(src_path, e))
            continue

        src_base = os.path.splitext(os.path.basename(src_path))[0]
        out_name = os.path.join(out_dir, f"{base_name}_{src_base}{base_ext}")
        print("[INFO] saving result to", out_name)
        cv2.imwrite(out_name, out)
        last_out = out

    # show result (optional) — only show when a single source was processed
    if not args.get("no_show") and last_out is not None and len(source_paths) == 1:
        cv2.imshow("Augmented", last_out)
        print("[INFO] press any key in image window to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
