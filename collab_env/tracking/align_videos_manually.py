"""
Script: align_videos_manually.py

Description:
    This script provides a complete pipeline for aligning RGB and thermal videos both spatially and temporally. It includes:
    - Cropping and rotating videos.
    - Spatial alignment using homography transformations.
    - Temporal alignment by adjusting frame offsets.
    - Saving the aligned videos.

Usage:
    Run this script from the command line with the required arguments:

    Example:
        python align_videos_manually.py \
            --rgb_video_path path/to/rgb_video.mp4 \
            --thermal_video_path path/to/thermal_video.mp4 \
            --output_dir_rgb path/to/output/rgb \
            --output_dir_thm path/to/output/thermal \
            --frame_size 640,480 \
            --max_frames 1000 \
            --warp_to rgb \
            --rotation_angle 0.0 \
            --skip_homography

Arguments:
    --rgb_video_path: Path to the RGB video file.
    --thermal_video_path: Path to the thermal video file.
    --output_dir_rgb: Directory to save the processed RGB video.
    --output_dir_thm: Directory to save the processed thermal video.
    --frame_size: Frame size for processing, specified as width,height (default: 640,480).
    --max_frames: Maximum number of frames to process (default: all frames).
    --warp_to: Target space for warping ('rgb' or 'thermal', default: 'rgb').
    --rotation_angle: Rotation angle for cropping (default: 0.0).
    --skip_homography: If set, only use translation for spatial alignment (no homography).

"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Helper Functions
def select_crop_on_concat(rgb_frame, thm_frame, frame_size=(640, 480), rotation_angle=None):
    """
    Allows the user to select a crop region on the RGB frame to match the thermal frame.

    Args:
        rgb_frame (np.ndarray): RGB frame.
        thm_frame (np.ndarray): Thermal frame.
        frame_size (tuple): Resized frame dimensions for display.
        rotation_angle (float, optional): Rotation angle for the crop.

    Returns:
        tuple: Crop rectangle (x, y, w, h) and rotation angle.
    """
    # Show resized for selection, but return crop in original frame coordinates
    rgb_resized = cv2.resize(rgb_frame, frame_size)
    thm_resized = cv2.resize(thm_frame, frame_size)
    concat = cv2.hconcat([rgb_resized, thm_resized])
    window_name = "Crop RGB (left) to match Thermal (right)"
    print("Draw a rectangle on the LEFT (RGB) image to crop. Press ENTER or SPACE when done.")
    r = cv2.selectROI(window_name, concat, showCrosshair=True, fromCenter=False)
    x, y, w, h = map(int, r)
    if (w==0 or h==0):
        print("⚠️ Please select a non-zero crop.")
        cv2.destroyWindow(window_name)
        return select_crop_on_concat(rgb_frame, thm_frame, frame_size, rotation_angle)
    if x + w > frame_size[0]:
        print("⚠️ Please select crop only within the left (RGB) image.")
        cv2.destroyWindow(window_name)
        return select_crop_on_concat(rgb_frame, thm_frame, frame_size, rotation_angle)
    cv2.destroyWindow(window_name)
    print("Selected crop (on RGB, resized):", (x, y, w, h))
    # Scale crop rect to original frame size
    scale_x = rgb_frame.shape[1] / frame_size[0]
    scale_y = rgb_frame.shape[0] / frame_size[1]
    x_orig = int(x * scale_x)
    y_orig = int(y * scale_y)
    w_orig = int(w * scale_x)
    h_orig = int(h * scale_y)
    print("Selected crop (on RGB, original):", (x_orig, y_orig, w_orig, h_orig))
    if rotation_angle is None:
        try:
            angle = float(input("Rotation angle (degrees, positive=CCW, negative=CW, 0 for none): "))
        except Exception:
            angle = 0.0
    else:
        angle = float(rotation_angle)
        print(f"Using provided rotation angle: {angle}")
    return (x_orig, y_orig, w_orig, h_orig), angle

def crop_and_rotate_video(input_path, output_path, crop_rect, angle, frame_size=None, max_frames=None):
    """
    Crops and rotates a video based on the provided crop rectangle and angle.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the cropped and rotated video.
        crop_rect (tuple): Crop rectangle (x, y, w, h).
        angle (float): Rotation angle.
        frame_size (tuple, optional): Output frame size.
        max_frames (int, optional): Maximum number of frames to process.

    Returns:
        None
    """
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)
    x, y, w, h = crop_rect
    if frame_size is None:
        frame_size = (w, h)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    for idx in tqdm(range(total), desc="Cropping/Rotating video"):
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y:y+h, x:x+w]
        if angle != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cropped = cv2.warpAffine(cropped, M, (w, h))
        cropped = cv2.resize(cropped, frame_size)
        out.write(cropped)
    cap.release()
    out.release()
    print(f"✅ Saved cropped/rotated video to {output_path}")

def select_points(img, window_name):
    """
    Allows the user to click points on the image. Returns a list of (x, y) points.

    Args:
        img (np.ndarray): Image to select points on.
        window_name (str): Name of the window for point selection.

    Returns:
        np.ndarray: Array of selected points.
    """
    points = []
    display_img = img.copy()
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            display_img[:] = img.copy()
            for px, py in points:
                cv2.circle(display_img, (px, py), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, display_img)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, click_event)
    print("Click at least 4 corresponding points in the image, then press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)

def get_translation(rgb_img, thm_img):
    """
    Allows the user to select a corresponding point in each image and returns a translation matrix.

    Args:
        rgb_img (np.ndarray): RGB image.
        thm_img (np.ndarray): Thermal image.

    Returns:
        np.ndarray: Translation matrix.
    """
    print("Click a corresponding point in the RGB image.")
    pt_rgb = select_points(rgb_img, "Select point in RGB")[0]
    print("Click the same point in the Thermal image.")
    pt_thm = select_points(thm_img, "Select point in Thermal")[0]
    tx = pt_thm[0] - pt_rgb[0]
    ty = pt_thm[1] - pt_rgb[1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    print(f"Translation: tx={tx}, ty={ty}")
    return M

def get_homography_with_translation(
    rgb_video_path,
    thermal_video_path,
    frame_size=(640, 480),
    warp_to='rgb',
    skip_homography=False,
    skip_translation=False
):
    # Load first frames
    rgb_cap = cv2.VideoCapture(str(rgb_video_path))
    thm_cap = cv2.VideoCapture(str(thermal_video_path))
    ret_rgb, frame_rgb = rgb_cap.read()
    ret_thm, frame_thm = thm_cap.read()
    rgb_cap.release()
    thm_cap.release()
    if not (ret_rgb and ret_thm):
        raise RuntimeError("Could not read frames for homography selection.")
    frame_rgb = cv2.resize(frame_rgb, frame_size)
    frame_thm = cv2.resize(frame_thm, frame_size)
    if len(frame_thm.shape) == 2 or frame_thm.shape[2] == 1:
        frame_thm = cv2.cvtColor(frame_thm, cv2.COLOR_GRAY2BGR)

    ###--- Translation step ---
    if skip_translation:
        print("Skipping translation step.")
        M = np.eye(2, 3, dtype=np.float32)  # Identity matrix for no translation
        translated_rgb = frame_rgb
        translated_thm = frame_thm
    else:
        M = get_translation(frame_rgb, frame_thm)
        if warp_to == 'rgb':
            translated_thm = cv2.warpAffine(frame_thm, M, frame_size)
            translated_rgb = frame_rgb
        else:
            translated_rgb = cv2.warpAffine(frame_rgb, M, frame_size)
            translated_thm = frame_thm

    if skip_homography:
        # Return translation as a 3x3 homography matrix
        H_translation = np.eye(3, dtype=np.float32)
        H_translation[:2, :] = M
        print("Returning translation-only homography.")
        return H_translation

    # --- Homography step ---
    alpha = 0.5
    if warp_to == 'rgb':
        overlay = cv2.addWeighted(frame_rgb, alpha, translated_thm, 1 - alpha, 0)
    else:
        overlay = cv2.addWeighted(translated_rgb, alpha, frame_thm, 1 - alpha, 0)
    window_name = "Overlay after translation: Adjust alpha with [ and ]. Click points, then press any key."
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            temp = overlay.copy()
            for px, py in points:
                cv2.circle(temp, (px, py), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, temp)

    cv2.imshow(window_name, overlay)
    cv2.setMouseCallback(window_name, click_event)
    print("Click at least 4 corresponding points in the overlay. Adjust alpha with [ and ]. Press any key when done.")

    while True:
        key = cv2.waitKey(0)
        if key == ord('['):
            alpha = max(0.0, alpha - 0.05)
            if warp_to == 'rgb':
                overlay = cv2.addWeighted(frame_rgb, alpha, translated_thm, 1 - alpha, 0)
            else:
                overlay = cv2.addWeighted(translated_rgb, alpha, frame_thm, 1 - alpha, 0)
            temp = overlay.copy()
            for px, py in points:
                cv2.circle(temp, (px, py), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, temp)
        elif key == ord(']'):
            alpha = min(1.0, alpha + 0.05)
            if warp_to == 'rgb':
                overlay = cv2.addWeighted(frame_rgb, alpha, translated_thm, 1 - alpha, 0)
            else:
                overlay = cv2.addWeighted(translated_rgb, alpha, frame_thm, 1 - alpha, 0)
            temp = overlay.copy()
            for px, py in points:
                cv2.circle(temp, (px, py), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, temp)
        else:
            break

    cv2.destroyWindow(window_name)
    pts_overlay = np.array(points, dtype=np.float32)

    # Now ask user to select the same points in the other image
    print(f"Now select the same {len(pts_overlay)} points in the other image, in the same order.")
    if warp_to == 'rgb':
        pts_ref = select_points(frame_rgb, "Select points in RGB")
    else:
        pts_ref = select_points(frame_thm, "Select points in Thermal")

    if pts_overlay.shape[0] >= 4 and pts_overlay.shape == pts_ref.shape:
        H, _ = cv2.findHomography(pts_overlay, pts_ref, cv2.RANSAC)
        print("Homography computed.")
        # Combine translation and homography
        H_full = np.dot(H, np.vstack([M, [0, 0, 1]]))
        return H_full
    else:
        raise RuntimeError("Invalid point selection for homography.")


def manual_temporal_alignment(rgb_video_path, thermal_video_path, H=None, frame_size=(640, 480), warp_to='rgb'):
    """
    Allows the user to manually align RGB and thermal videos temporally.

    Args:
        rgb_video_path (str): Path to the RGB video.
        thermal_video_path (str): Path to the thermal video.
        H (np.ndarray, optional): Homography matrix for spatial alignment.
        frame_size (tuple): Frame size for processing.
        warp_to (str): Target space for warping ('rgb' or 'thermal').

    Returns:
        int: Frame offset for temporal alignment.
    """
    import cv2
    print(f"DEBUG: RGB path: {rgb_video_path}")
    print(f"DEBUG: THERMAL path: {thermal_video_path}")
    print(f"DEBUG: H is None: {H is None}")
    rgb_cap = cv2.VideoCapture(str(rgb_video_path))
    thm_cap = cv2.VideoCapture(str(thermal_video_path))
    # ...rest of your code...
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thm_total = int(thm_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_idx = 0
    thm_idx = 0
    alpha = 0.5
    window_name = "Temporal Alignment (Overlay)"
    print("Use A/D to move RGB, W/S to move Thermal. Press SPACE when aligned. ESC to cancel.")

    while True:
        rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_idx)
        thm_cap.set(cv2.CAP_PROP_POS_FRAMES, thm_idx)
        ret_rgb, frame_rgb = rgb_cap.read()
        ret_thm, frame_thm = thm_cap.read()
        if not (ret_rgb and ret_thm):
            print("Could not read frames for alignment.")
            break

        frame_rgb = cv2.resize(frame_rgb, frame_size)
        frame_thm = cv2.resize(frame_thm, frame_size)
        if len(frame_thm.shape) == 2 or frame_thm.shape[2] == 1:
            frame_thm = cv2.cvtColor(frame_thm, cv2.COLOR_GRAY2BGR)

        # --- Always spatially align before overlay ---
        if H is not None:
            if warp_to == 'thermal':
                # Warp RGB to thermal
                if H.shape == (3, 3):
                    frame_rgb_aligned = cv2.warpPerspective(frame_rgb, H, frame_size)
                elif H.shape == (2, 3):
                    frame_rgb_aligned = cv2.warpAffine(frame_rgb, H, frame_size)
                else:
                    raise ValueError("Unknown transformation matrix shape")
                overlay = cv2.addWeighted(frame_rgb_aligned, alpha, frame_thm, 1 - alpha, 0)
            elif warp_to == 'rgb':
                # Warp THERMAL to RGB
                if H.shape == (3, 3):
                    frame_thm_aligned = cv2.warpPerspective(frame_thm, H, frame_size)
                elif H.shape == (2, 3):
                    frame_thm_aligned = cv2.warpAffine(frame_thm, H, frame_size)
                else:
                    raise ValueError("Unknown transformation matrix shape")
                overlay = cv2.addWeighted(frame_rgb, alpha, frame_thm_aligned, 1 - alpha, 0)
            else:
                raise ValueError("warp_to must be 'thermal' or 'rgb'")
        else:
            overlay = cv2.addWeighted(frame_rgb, alpha, frame_thm, 1 - alpha, 0)

        cv2.putText(overlay, f"RGB idx: {rgb_idx} | THM idx: {thm_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("Cancelled.")
            cv2.destroyWindow(window_name)
            return 0
        elif key == ord(' '):  # SPACE
            print(f"Selected RGB frame {rgb_idx}, Thermal frame {thm_idx}")
            cv2.destroyWindow(window_name)
            return thm_idx - rgb_idx
        elif key == ord('a'):  # A for left
            rgb_idx = max(0, rgb_idx - 1)
        elif key == ord('d'):  # D for right
            rgb_idx = min(rgb_total-1, rgb_idx + 1)
        elif key == ord('w'):  # W for up
            thm_idx = max(0, thm_idx - 1)
        elif key == ord('s'):  # S for down
            thm_idx = min(thm_total-1, thm_idx + 1)
        cv2.waitKey(3)

    cv2.destroyAllWindows()

### steps to run the full alignment pipeline

def step1_crop_and_prepare(rgb_video_path, thermal_video_path, output_dir_rgb, output_dir_thm, frame_size, max_frames):
    aligned_frames_root = os.path.dirname(os.path.dirname(output_dir_rgb))
    os.makedirs(aligned_frames_root, exist_ok=True)
    os.makedirs(output_dir_rgb, exist_ok=True)
    os.makedirs(output_dir_thm, exist_ok=True)

    rgb_cap = cv2.VideoCapture(str(rgb_video_path))
    thm_cap = cv2.VideoCapture(str(thermal_video_path))
    ret_rgb, rgb_sample = rgb_cap.read()
    ret_thm, thm_sample = thm_cap.read()
    if not (ret_rgb and ret_thm):
        raise RuntimeError("Could not read videos for preview/cropping.")

    crop_rect, angle = select_crop_on_concat(rgb_sample, thm_sample, frame_size=frame_size)
    cropped_rgb_path = os.path.join(output_dir_rgb, "cropped_rgb.mp4")
    crop_and_rotate_video(rgb_video_path, cropped_rgb_path, crop_rect, angle, frame_size=frame_size, max_frames=max_frames)
    print(f"✅ Saved cropped RGB video to {cropped_rgb_path}")
    rgb_cap.release()
    thm_cap.release()
    return cropped_rgb_path

def step2_spatial_alignment(cropped_rgb_path, thermal_video_path, frame_size, warp_to="thermal", skip_homography=False, skip_translation=False):
    H = get_homography_with_translation(
        cropped_rgb_path,
        thermal_video_path,
        frame_size=frame_size,
        warp_to=warp_to,
        skip_homography=skip_homography,
        skip_translation=skip_translation
    )
    print("✅ Transformation matrix:\n", H)
    return H

def save_warped_video(input_video_path, output_video_path, H, frame_size, max_frames=None):
    if H.shape == (3, 3):
        H_inv = np.linalg.inv(H)
    elif H.shape == (2, 3):
        A = H[:, :2]
        b = H[:, 2]
        A_inv = np.linalg.inv(A)
        b_inv = -A_inv @ b
        H_inv = np.hstack([A_inv, b_inv.reshape(2, 1)])
    else:
        raise ValueError("Unknown transformation matrix shape")

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total_frames, max_frames) if max_frames else total_frames
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for i in tqdm(range(n_frames), desc="Warping thermal video"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        if H.shape == (3, 3):
            warped = cv2.warpPerspective(frame, H_inv, frame_size)
        else:
            warped = cv2.warpAffine(frame, H_inv, frame_size)
        out.write(warped)
    cap.release()
    out.release()
    print(f"✅ Warped thermal video saved to {output_video_path}")

def step3_temporal_alignment(cropped_rgb_path, warped_thermal_path, frame_size, warp_to="rgb"):
    frame_offset = manual_temporal_alignment(
        cropped_rgb_path,
        warped_thermal_path,
        H=None,
        frame_size=frame_size,
        warp_to=warp_to
    )
    print("✅ Frame offset (temporal alignment):", frame_offset)
    return frame_offset

def save_temporally_adjusted_video(input_video_path, output_video_path, frame_offset, frame_size):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    start_frame = max(0, frame_offset) if frame_offset > 0 else abs(frame_offset)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in tqdm(range(total_frames - start_frame), desc="Saving adjusted RGB video"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Adjusted RGB video saved to {output_video_path}")

def align_videos(rgb_video_path, thermal_video_path, output_dir_rgb, output_dir_thm,
                 frame_size=(640, 480), max_frames=None, warp_to="thermal", rotation_angle=0.0, skip_homography=False, skip_translation=True):

    cropped_rgb_path = step1_crop_and_prepare(rgb_video_path, thermal_video_path, output_dir_rgb, output_dir_thm, frame_size, max_frames)

    H = step2_spatial_alignment(cropped_rgb_path, thermal_video_path, frame_size, warp_to, skip_homography, skip_translation=skip_translation)

    warped_thermal_path = os.path.join(output_dir_thm, "warped_thermal.mp4")
    save_warped_video(thermal_video_path, warped_thermal_path, H, frame_size, max_frames)

    frame_offset = step3_temporal_alignment(cropped_rgb_path, warped_thermal_path, frame_size, warp_to)

    adjusted_rgb_path = os.path.join(output_dir_rgb, "adjusted_rgb.mp4")
    save_temporally_adjusted_video(cropped_rgb_path, adjusted_rgb_path, frame_offset, frame_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_video_path", type=str, required=True)
    parser.add_argument("--thermal_video_path", type=str, required=True)
    parser.add_argument("--output_dir_rgb", type=str, required=True)
    parser.add_argument("--output_dir_thm", type=str, required=True)
    parser.add_argument("--frame_size", type=str, default="640,480")
    parser.add_argument("--max_frames", type=str, default="")
    parser.add_argument("--warp_to", type=str, default="rgb")
    parser.add_argument("--rotation_angle", type=float, default=0.0)
    parser.add_argument("--skip_homography", action="store_true", help="If set, only use translation (no homography)")
    args = parser.parse_args()

    frame_size = tuple(map(int, args.frame_size.split(",")))
    max_frames = int(args.max_frames) if args.max_frames.strip() else None

    align_videos(
        args.rgb_video_path,
        args.thermal_video_path,
        args.output_dir_rgb,
        args.output_dir_thm,
        frame_size=frame_size,
        max_frames=max_frames,
        warp_to=args.warp_to,
        rotation_angle=args.rotation_angle,
        skip_homography=args.skip_homography
    )
