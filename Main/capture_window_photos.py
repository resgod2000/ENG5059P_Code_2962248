"""
Capture the contents of the "RealSense RGB Image" window once per second and
store the screenshots as JPG files inside the ./dataset_cone/image directory.
"""

import os
import time
import ctypes
from ctypes import wintypes

import cv2
import numpy as np

WINDOW_TITLE = "RealSense RGB Image"
PHOTO_DIR = os.path.join(os.path.dirname(__file__), "dataset_cone", "image")
CAPTURE_PERIOD = 2.0  # seconds

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

SW_SHOW = 5
PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def ensure_photo_dir() -> None:
    os.makedirs(PHOTO_DIR, exist_ok=True)


def get_next_index() -> int:
    """Return the next numeric filename index based on existing JPG files."""
    max_index = 0
    for name in os.listdir(PHOTO_DIR):
        base, ext = os.path.splitext(name)
        if ext.lower() == ".jpg" and base.isdigit():
            num = int(base)
            if num > max_index:
                max_index = num
    return max_index + 1


def capture_window(hwnd: int) -> np.ndarray | None:
    if not hwnd:
        return None

    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    window_dc = user32.GetDC(hwnd)
    if not window_dc:
        return None

    mem_dc = gdi32.CreateCompatibleDC(window_dc)
    if not mem_dc:
        user32.ReleaseDC(hwnd, window_dc)
        return None

    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    if not bitmap:
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, window_dc)
        return None

    gdi32.SelectObject(mem_dc, bitmap)

    success = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
    if not success:
        # Fallback to BitBlt if PrintWindow fails
        user32.ShowWindow(hwnd, SW_SHOW)
        gdi32.BitBlt(mem_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY)

    bmp_info = BITMAPINFOHEADER()
    bmp_info.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmp_info.biWidth = width
    bmp_info.biHeight = -height  # top-down DIB
    bmp_info.biPlanes = 1
    bmp_info.biBitCount = 32
    bmp_info.biCompression = 0  # BI_RGB

    buffer_size = width * height * 4
    buffer = (ctypes.c_ubyte * buffer_size)()

    bits = gdi32.GetDIBits(mem_dc, bitmap, 0, height, ctypes.byref(buffer), ctypes.byref(bmp_info), DIB_RGB_COLORS)

    frame = None
    if bits != 0:
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, window_dc)

    return frame


def main() -> None:
    ensure_photo_dir()
    counter = get_next_index()

    print(f"Waiting for window titled '{WINDOW_TITLE}'...")
    while True:
        hwnd = user32.FindWindowW(None, WINDOW_TITLE)
        if hwnd:
            break
        time.sleep(0.5)

    print("Window found. Starting capture...")
    try:
        while True:
            frame = capture_window(hwnd)
            if frame is not None:
                filename = f"{counter}.jpg"
                filepath = os.path.join(PHOTO_DIR, filename)
                cv2.imwrite(filepath, frame)
                print(f"Saved {filepath}")
                counter += 1
            else:
                print("Warning: Unable to capture window frame.")
            time.sleep(CAPTURE_PERIOD)
    except KeyboardInterrupt:
        print("Capture interrupted by user.")


if __name__ == "__main__":
    main()
