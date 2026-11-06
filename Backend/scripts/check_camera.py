"""Check that a camera source is reachable before starting SHIELD.

    python scripts/check_camera.py http://192.168.1.101:8080/video
    python scripts/check_camera.py 0                     # laptop webcam

Prints resolution and measured FPS, or a diagnosis if it cannot connect.
"""
import ipaddress
import socket
import sys
import time
from urllib.parse import urlparse

import cv2


def diagnose(url: str) -> None:
    host, port = urlparse(url).hostname, urlparse(url).port or 80
    try:
        ip = ipaddress.ip_address(host)
        if ip in ipaddress.ip_network("192.0.0.0/24"):
            print(f"  {host} is a mobile-data (carrier) address, not reachable from the laptop.\n"
                  "  Put phone and laptop on the same Wi-Fi, or connect the laptop to the phone's\n"
                  "  hotspot, then use the new IPv4 address IP Webcam shows.")
            return
    except ValueError:
        pass
    try:
        socket.create_connection((host, port), timeout=4).close()
        print(f"  {host}:{port} answers, but no video at this path. Use the app's /video URL.")
    except OSError:
        print(f"  Cannot reach {host}:{port}. Check: same network? server started in the app?\n"
              "  Some college/office Wi-Fi blocks device-to-device traffic; use the phone hotspot.")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    is_url = "://" in src
    if is_url:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000,
                                                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000])
    else:
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"FAIL: cannot open {src}")
        if is_url:
            diagnose(src)
        return 1
    n, t, shape = 0, time.time(), None
    while time.time() - t < 3:
        ok, frame = cap.read()
        if not ok:
            break
        n, shape = n + 1, frame.shape
    cap.release()
    if not n:
        print(f"FAIL: connected to {src} but received no frames")
        return 1
    print(f"OK: {src}  {shape[1]}x{shape[0]}  ~{n / (time.time() - t):.0f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
