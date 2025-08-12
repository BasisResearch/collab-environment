"""
Script: alignment_gui.py

Description:
    This script provides a complete pipeline for aligning RGB and thermal videos both spatially and temporally. It includes:
    - Cropping and rotating videos.
    - Spatial alignment using homography transformations.
    - Temporal alignment by adjusting frame offsets.
    - Saving the aligned videos.

Usage:
    Run this script from the command line with the required arguments:

    Example:
        python alignment_gui.py \
            --rgb_video_path path/to/rgb_video.mp4 \
            --thermal_video_path path/to/thermal_video.mp4 \
            --output_dir_rgb path/to/output/rgb \
            --output_dir_thm path/to/output/thermal \
            --frame_size 640,480 \
            --max_frames 1000 \
            --warp_to rgb \
            --rotation_angle 0.0 \
            --skip_homography \
            --skip_translation

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
    --skip_translation: If set, only use homography (no translation).

"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


# Helper Functions
def select_crop_on_concat(
    rgb_frame, thm_frame, frame_size=(640, 480), rotation_angle=None
):
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
    print(
        "Draw a rectangle on the LEFT (RGB) image to crop. Press ENTER or SPACE when done."
    )
    r = cv2.selectROI(window_name, concat, showCrosshair=True, fromCenter=False)
    x, y, w, h = map(int, r)
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
            angle = float(
                input(
                    "Rotation angle (degrees, positive=CCW, negative=CW, 0 for none): "
                )
            )
        except Exception:
            angle = 0.0
    else:
        angle = float(rotation_angle)
        print(f"Using provided rotation angle: {angle}")
    return (x_orig, y_orig, w_orig, h_orig), angle


def crop_and_rotate_video(
    input_path, output_path, crop_rect, angle, frame_size=None, max_frames=None
):
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
    out = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc("M", "P", "4", "V"), fps, frame_size
    )

    for idx in tqdm(range(total), desc="Cropping/Rotating video"):
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y : y + h, x : x + w]
        if angle != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cropped = cv2.warpAffine(cropped, M, (w, h))
        cropped = cv2.resize(cropped, frame_size)
        out.write(cropped)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Adjusted RGB video saved to {output_path}")


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
    print(
        "Click at least 4 corresponding points in the image, then press any key to continue."
    )
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
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    print(f"Translation: tx={tx}, ty={ty}")
    return M


def get_homography_with_translation(
    rgb_video_path,
    thermal_video_path,
    frame_size=(640, 480),
    warp_to="rgb",
    skip_homography=False,
    skip_translation=False,
):
    """
    Allows the user to align RGB and thermal videos spatially using translation and homography.

    Args:
        rgb_video_path (str): Path to the RGB video.
        thermal_video_path (str): Path to the thermal video.
        frame_size (tuple): Frame size for processing.
        warp_to (str): Target space for warping ('rgb' or 'thermal').
        skip_homography (bool): If True, skip homography step.
        skip_translation (bool): If True, skip translation step.

    Returns:
        np.ndarray: Transformation matrix (homography or affine).

    Note:
        The returned matrix always maps points from the source video to the target video.
        For example, if warp_to='rgb', the matrix maps thermal → rgb.
        The saving function will always invert the matrix before warping, as required by OpenCV.
    """
    import tkinter as tk

    # Instructions popup
    def show_instructions():
        instructions = (
            "1. Select corresponding points on the RGB and thermal videos displayed side by side.\n"
            "2. Use at least 4 points for accurate alignment.\n"
            "3. Press any key to confirm the selection and compute the transformation.\n"
            "4. Press ESC to cancel the process."
        )
        popup = tk.Tk()
        popup.title("Spatial Alignment Instructions")
        tk.Label(popup, text=instructions, justify="left", wraplength=400).pack(pady=10)
        tk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)
        popup.mainloop()

    show_instructions()

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

    # Display RGB and thermal videos side by side
    concat = cv2.hconcat([frame_rgb, frame_thm])
    window_name = "Select Corresponding Points (RGB: Left, Thermal: Right)"
    points_src = []
    points_dst = []
    is_src_turn = True

    # Decide which is source and which is target based on warp_to
    if warp_to == "rgb":
        src_label = "Thermal"
        dst_label = "RGB"
        src_frame = frame_thm
        dst_frame = frame_rgb
        src_offset = frame_rgb.shape[1]
    else:
        src_label = "RGB"
        dst_label = "Thermal"
        src_frame = frame_rgb
        dst_frame = frame_thm
        src_offset = 0

    def click_event(event, x, y, flags, param):
        nonlocal is_src_turn
        if event == cv2.EVENT_LBUTTONDOWN:
            if is_src_turn and x >= src_offset and x < src_offset + src_frame.shape[1]:
                points_src.append((x - src_offset, y))
                is_src_turn = False
            elif not is_src_turn and x < dst_frame.shape[1]:
                points_dst.append((x, y))
                is_src_turn = True

            temp = concat.copy()
            for px, py in points_src:
                cv2.circle(temp, (px + src_offset, py), 5, (0, 255, 0), -1)
            for px, py in points_dst:
                cv2.circle(temp, (px, py), 5, (255, 0, 0), -1)
            cv2.imshow(window_name, temp)

    cv2.imshow(window_name, concat)
    cv2.setMouseCallback(window_name, click_event)
    print(
        f"Click corresponding points alternately in the {src_label} (right) and {dst_label} (left) videos. Press ENTER when done."
    )

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # ENTER key
            if len(points_src) < 4 or len(points_dst) < 4:
                print(
                    "At least 4 pairs of points are required for homography computation."
                )
            else:
                break

    cv2.destroyWindow(window_name)

    if len(points_src) != len(points_dst):
        raise RuntimeError(
            "Mismatch in the number of points selected for source and target videos."
        )

    pts_src = np.array(points_src, dtype=np.float32)
    pts_dst = np.array(points_dst, dtype=np.float32)

    # Compute homography or translation
    if not skip_homography:
        H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
        print("Homography computed.")
    elif not skip_translation:
        tx = np.mean(pts_dst[:, 0] - pts_src[:, 0])
        ty = np.mean(pts_dst[:, 1] - pts_src[:, 1])
        H = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        print(f"Translation computed: tx={tx}, ty={ty}")
    else:
        raise ValueError("Both homography and translation are skipped.")

    print("Transformation matrix (source → target):\n", H)
    return H


def save_warped_video(
    input_video_path, output_video_path, H, frame_size, max_frames=None
):
    """
    Save a warped video by applying a transformation matrix (H).

    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path to save the warped video.
        H (np.ndarray): Transformation matrix (2x3 or 3x3), mapping source → target.
        frame_size (tuple): Frame size (width, height).
        max_frames (int, optional): Maximum number of frames to process. Defaults to None (process all frames).

    Note:
        OpenCV warping functions require the inverse transformation (target → source).
        This function always inverts the matrix before warping, so you never need to change any settings.
    """
    # Invert H for OpenCV warping (target → source)
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total_frames, max_frames) if max_frames else total_frames
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for i in tqdm(range(n_frames), desc="Warping video"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        if H_inv.shape == (3, 3):
            warped = cv2.warpPerspective(frame, H_inv, frame_size)
        elif H_inv.shape == (2, 3):
            warped = cv2.warpAffine(frame, H_inv, frame_size)
        out.write(warped)
    cap.release()
    out.release()
    print(f"Warped video saved to {output_video_path}")


def manual_temporal_alignment(
    rgb_video_path, thermal_video_path, H=None, frame_size=(640, 480), warp_to="rgb"
):
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
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    thm_total = int(thm_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_idx = 0
    thm_idx = 0
    alpha = 0.5
    window_name = "Temporal Alignment (Overlay)"

    print(f"RGB Total Frames: {rgb_total}, Thermal Total Frames: {thm_total}")
    print(
        "Use A/D to move RGB, W/S to move Thermal. Adjust Alpha with [ and ]. Press SPACE when aligned. ESC to cancel."
    )

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
            if warp_to == "thermal":
                # Warp RGB to thermal
                if H.shape == (3, 3):
                    frame_rgb_aligned = cv2.warpPerspective(frame_rgb, H, frame_size)
                elif H.shape == (2, 3):
                    frame_rgb_aligned = cv2.warpAffine(frame_rgb, H, frame_size)
                else:
                    raise ValueError("Unknown transformation matrix shape")
                overlay = cv2.addWeighted(
                    frame_rgb_aligned, alpha, frame_thm, 1 - alpha, 0
                )
            elif warp_to == "rgb":
                # Warp THERMAL to RGB
                if H.shape == (3, 3):
                    frame_thm_aligned = cv2.warpPerspective(frame_thm, H, frame_size)
                elif H.shape == (2, 3):
                    frame_thm_aligned = cv2.warpAffine(frame_thm, H, frame_size)
                else:
                    raise ValueError("Unknown transformation matrix shape")
                overlay = cv2.addWeighted(
                    frame_rgb, alpha, frame_thm_aligned, 1 - alpha, 0
                )
            else:
                raise ValueError("warp_to must be 'thermal' or 'rgb'")
        else:
            overlay = cv2.addWeighted(frame_rgb, alpha, frame_thm, 1 - alpha, 0)

        cv2.putText(
            overlay,
            f"RGB idx: {rgb_idx}/{rgb_total} | THM idx: {thm_idx}/{thm_total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("Cancelled.")
            cv2.destroyWindow(window_name)
            return 0
        elif key == ord(" "):  # SPACE
            print(f"Selected RGB frame {rgb_idx}, Thermal frame {thm_idx}")
            cv2.destroyWindow(window_name)
            return thm_idx - rgb_idx
        elif key == ord("a"):  # A for left
            rgb_idx = max(0, rgb_idx - 1)
        elif key == ord("d"):  # D for right
            rgb_idx = min(rgb_total - 1, rgb_idx + 1)
        elif key == ord("w"):  # W for up
            thm_idx = max(0, thm_idx - 1)
        elif key == ord("s"):  # S for down
            thm_idx = min(thm_total - 1, thm_idx + 1)
        elif key == ord("["):  # Decrease alpha
            alpha = max(0.0, alpha - 0.1)
            print(f"Alpha decreased to {alpha:.1f}")
        elif key == ord("]"):  # Increase alpha
            alpha = min(1.0, alpha + 0.1)
            print(f"Alpha increased to {alpha:.1f}")
        cv2.waitKey(3)
    cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()  # Ensure all windows are closed
    rgb_cap.release()
    thm_cap.release()


### steps to run the full alignment pipeline


def step1_crop_and_prepare(
    rgb_video_path,
    thermal_video_path,
    output_dir_rgb,
    output_dir_thm,
    frame_size,
    max_frames,
):
    aligned_frames_root = os.path.dirname(os.path.dirname(output_dir_rgb))
    os.makedirs(aligned_frames_root, exist_ok=True)
    os.makedirs(output_dir_rgb, exist_ok=True)
    os.makedirs(output_dir_thm, exist_ok=True)

    rgb_cap = cv2.VideoCapture(str(rgb_video_path))
    thm_cap = cv2.VideoCapture(str(thermal_video_path))
    ret_rgb, rgb_sample = rgb_cap.read()
    ret_thm, thm_sample = thm_cap.read()
    rgb_cap.release()
    thm_cap.release()

    if not (ret_rgb and ret_thm):
        raise RuntimeError("Could not read videos for preview/cropping.")

    crop_rect, angle = cropping_step(rgb_sample, thm_sample, frame_size=frame_size)

    if crop_rect is None:
        raise RuntimeError("Cropping was not completed successfully.")

    cropped_rgb_path = os.path.join(output_dir_rgb, "cropped_rgb.mp4")
    crop_and_rotate_video(
        rgb_video_path,
        cropped_rgb_path,
        crop_rect,
        angle,
        frame_size=frame_size,
        max_frames=max_frames,
    )
    print(f"✅ Saved cropped RGB video to {cropped_rgb_path}")

    return cropped_rgb_path


def step2_spatial_alignment(
    cropped_rgb_path,
    thermal_video_path,
    frame_size,
    warp_to="thermal",
    skip_homography=False,
    skip_translation=False,
):
    H = get_homography_with_translation(
        cropped_rgb_path,
        thermal_video_path,
        frame_size=frame_size,
        warp_to=warp_to,
        skip_homography=skip_homography,
        skip_translation=skip_translation,
    )
    print("✅ Transformation matrix:\n", H)
    return H


def step3_temporal_alignment(
    cropped_rgb_path, warped_thermal_path, frame_size, warp_to="rgb"
):
    frame_offset = manual_temporal_alignment(
        cropped_rgb_path,
        warped_thermal_path,
        H=None,
        frame_size=frame_size,
        warp_to=warp_to,
    )
    print("✅ Frame offset (temporal alignment):", frame_offset)
    return frame_offset


def save_temporally_adjusted_video(
    input_video_path, output_video_path, frame_offset, frame_size
):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "V")
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


def align_videos(
    rgb_video_path,
    thermal_video_path,
    output_dir_rgb,
    output_dir_thm,
    frame_size=(640, 480),
    max_frames=None,
    warp_to="thermal",
    rotation_angle=0.0,
    skip_homography=False,
    skip_translation=True,
):
    cropped_rgb_path = step1_crop_and_prepare(
        rgb_video_path,
        thermal_video_path,
        output_dir_rgb,
        output_dir_thm,
        frame_size,
        max_frames,
    )

    H = step2_spatial_alignment(
        cropped_rgb_path,
        thermal_video_path,
        frame_size,
        warp_to,
        skip_homography,
        skip_translation=skip_translation,
    )

    warped_thermal_path = os.path.join(output_dir_thm, "warped_thermal.mp4")
    save_warped_video(
        thermal_video_path, warped_thermal_path, H, frame_size, max_frames
    )

    frame_offset = step3_temporal_alignment(
        cropped_rgb_path, warped_thermal_path, frame_size, warp_to
    )

    adjusted_rgb_path = os.path.join(output_dir_rgb, "adjusted_rgb.mp4")
    adjusted_thermal_path = os.path.join(output_dir_thm, "adjusted_thermal.mp4")
    save_temporally_adjusted_video(
        cropped_rgb_path, adjusted_rgb_path, frame_offset, frame_size
    )

    save_temporally_adjusted_video(
        warped_thermal_path, adjusted_thermal_path, frame_offset, frame_size
    )


def cropping_step(rgb_frame, thm_frame, frame_size=(640, 480)):
    """
    Displays RGB and thermal videos side by side for cropping with instructions, dynamic rotation slider, and error handling.

    Args:
        rgb_frame (np.ndarray): RGB frame for cropping.
        thm_frame (np.ndarray): Thermal frame for reference.
        frame_size (tuple): Frame size for display.

    Returns:
        tuple: Crop rectangle (x, y, w, h) and rotation angle.
    """
    import tkinter as tk
    from tkinter import messagebox

    # Instructions popup
    def show_instructions():
        instructions = (
            "1. Adjust the rotation angle using the slider, then press ENTER to move to cropping step.\n"
            "2. Draw a rectangle on the RGB video (left) to crop, then press ENTER to confirm the crop.\n"
            "3. Wait for frame writing process to complete.\n"
            "Note: Press ESC to cancel the process."
        )
        popup = tk.Tk()
        popup.title("Instructions")
        tk.Label(popup, text=instructions, justify="left", wraplength=400).pack(pady=10)
        tk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)
        popup.mainloop()

    show_instructions()

    def rotate_image(img, angle):
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    # Resize frames for display
    rgb_resized = cv2.resize(rgb_frame, frame_size)
    thm_resized = cv2.resize(thm_frame, frame_size)
    concat = cv2.hconcat([rgb_resized, thm_resized])

    # --- Step 1: Trackbar rotation ---
    def on_trackbar(val):
        angle = cv2.getTrackbarPos("Angle", "Rotate") - 180
        rotated = rotate_image(rgb_resized, angle)
        updated_concat = cv2.hconcat([rotated, thm_resized])
        cv2.imshow("Rotate", updated_concat)

    cv2.namedWindow("Rotate")
    cv2.createTrackbar("Angle", "Rotate", 180, 360, on_trackbar)
    on_trackbar(180)

    print("🌀 Adjust rotation. Press any key when satisfied...")
    cv2.waitKey(0)
    angle = cv2.getTrackbarPos("Angle", "Rotate") - 180
    cv2.destroyWindow("Rotate")
    print(f"✅ Selected angle: {angle}°")

    # --- Step 2: Cropping ---
    window_name = "Crop RGB (left) to match Thermal (right)"
    cv2.imshow(window_name, concat)

    while True:
        try:
            print(
                "Draw a rectangle on the LEFT (RGB) image to crop. Press ENTER or SPACE when done."
            )
            r = cv2.selectROI(window_name, concat, showCrosshair=True, fromCenter=False)
            x, y, w, h = map(int, r)

            if x + w > frame_size[0]:
                messagebox.showerror(
                    "Error", "Please select a crop only within the left (RGB) image."
                )
                continue

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

            return (x_orig, y_orig, w_orig, h_orig), angle

        except KeyboardInterrupt:
            print("Cropping process terminated by user.")
            cv2.destroyAllWindows()
            raise

        except Exception as e:
            print("Error during cropping:", e)
            messagebox.showerror(
                "Error", "Cropping was interrupted or invalid. Please try again."
            )
            cv2.destroyWindow(window_name)
            return None, None

        finally:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("ESC key pressed. Terminating cropping process.")
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("Cropping process terminated by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_video_path", type=str, required=True)
    parser.add_argument("--thermal_video_path", type=str, required=True)
    parser.add_argument("--output_dir_rgb", type=str, required=True)
    parser.add_argument("--output_dir_thm", type=str, required=True)
    parser.add_argument("--frame_size", type=str, default="640,480")
    parser.add_argument("--max_frames", type=str, default="100")
    parser.add_argument("--warp_to", type=str, default="rgb")
    parser.add_argument("--rotation_angle", type=float, default=0.0)
    parser.add_argument(
        "--skip_homography",
        action="store_true",
        help="If set, only use translation (no homography)",
    )
    parser.add_argument(
        "--skip_translation",
        action="store_true",
        help="If set, only use homography (no translation)",
    )
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
        skip_homography=args.skip_homography,
        skip_translation=args.skip_translation,
    )
    print("✅ Alignment process completed successfully.")
