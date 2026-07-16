"""Camera/video rock-paper-scissors with MediaPipe hand landmarks.

Run a camera game:
    uv run --offline --no-project python rps_game.py

Run the supplied regression video without opening a window:
    uv run --offline --no-project python rps_game.py --video test.mov --headless
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Deque, Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CHOICES = ("rock", "paper", "scissors")
DISPLAY_NAME = {"rock": "石頭", "paper": "布", "scissors": "剪刀", "unknown": "無法辨識"}
CHINESE_FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"

# MediaPipe hand-landmark indices, ordered wrist -> fingertip.
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


def load_choice_images() -> dict[str, np.ndarray]:
    """Load the supplied computer-choice artwork exactly once."""
    images: dict[str, np.ndarray] = {}
    for choice in CHOICES:
        path = ROOT / f"{choice}.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Missing computer image: {path}")
        images[choice] = image
    return images


def classify_landmarks(hand) -> str:
    """Classify RPS from landmarks when the built-in gesture label is unavailable.

    The comparison uses distances from the wrist, normalized by palm width, so it
    works with either left or right hands and is resilient to camera distance.
    """
    points = np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32)
    palm_width = max(float(np.linalg.norm(points[5] - points[17])), 1e-5)
    extended = []
    for _, mcp, pip, tip in FINGERS:
        tip_distance = np.linalg.norm(points[tip] - points[0]) / palm_width
        pip_distance = np.linalg.norm(points[pip] - points[0]) / palm_width
        # A straight raised finger extends clearly beyond its PIP joint.
        extended.append(tip_distance > pip_distance + 0.34)

    # A sideways hand can make wrist-to-tip distances misleading.  A compact
    # palm is still easy to distinguish from an open palm using each finger's
    # MCP-to-tip span.
    spans = [np.linalg.norm(points[tip] - points[mcp]) / palm_width for _, mcp, _, tip in FINGERS[1:]]
    index, middle, ring, pinky = extended[1:]
    raised = sum((index, middle, ring, pinky))
    if min(spans) > 0.40 and max(spans) - min(spans) < 0.16:
        return "paper"
    if spans[0] > 0.40 and min(spans[1:]) < 0.35:
        return "scissors"
    if raised >= 3:
        return "paper"
    if index and middle and not ring and not pinky:
        return "scissors"
    if raised == 0:
        return "rock"
    return "unknown"


def outcome(player: str, computer: str) -> str:
    if player == computer:
        return "平手"
    wins = {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}
    return "你贏了！" if (player, computer) in wins else "電腦贏了！"


def gesture_from_result(result) -> str:
    """Map MediaPipe canned gestures; use landmark geometry as a safe fallback."""
    if not result.hand_landmarks:
        return "unknown"
    label = result.gestures[0][0].category_name if result.gestures and result.gestures[0] else ""
    # A fist held edge-on during the shake is occasionally labelled Thumb_Up
    # by the canned classifier; the RPS landmark fallback treats that pose as
    # the closed-fist preparation used by this game.
    labels = {"Open_Palm": "paper", "Closed_Fist": "rock", "Victory": "scissors", "Thumb_Up": "rock"}
    return labels.get(label, classify_landmarks(result.hand_landmarks[0]))


def draw_landmarks(frame: np.ndarray, hand) -> None:
    h, w = frame.shape[:2]
    xy = [(int(p.x * w), int(p.y * h)) for p in hand]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, xy[start], xy[end], (80, 220, 80), 2, cv2.LINE_AA)
    for point in xy:
        cv2.circle(frame, point, 4, (30, 90, 255), -1, cv2.LINE_AA)


def paste_choice(frame: np.ndarray, image: np.ndarray) -> None:
    """Alpha-composite the computer image into the upper-right of the frame."""
    max_w = max(1, frame.shape[1] // 3)
    scale = min(max_w / image.shape[1], (frame.shape[0] * 0.48) / image.shape[0], 1.0)
    overlay = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    x, y = frame.shape[1] - w - 18, 18
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        frame[y:y + h, x:x + w] = (overlay[:, :, :3] * alpha + frame[y:y + h, x:x + w] * (1 - alpha)).astype(np.uint8)
    else:
        frame[y:y + h, x:x + w] = overlay[:, :, :3]


def text(frame: np.ndarray, value: str, xy: tuple[int, int], scale: float = 0.65, color=(255, 255, 255)) -> None:
    # Hershey cannot render CJK glyphs on all OpenCV builds; concise English UI
    # keeps the program portable while result values remain clear in the console.
    cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


@lru_cache(maxsize=4)
def chinese_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(CHINESE_FONT, size)


def draw_status_panel(frame: np.ndarray, gesture: str,
                      result: Optional[tuple[str, str, str]], throw_status: Optional[str]) -> None:
    """Draw a high-contrast, translucent status area in the upper-left."""
    panel = frame.copy()
    cv2.rectangle(panel, (12, 12), (760, 275), (255, 255, 255), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((30, 26), f"目前手勢：{DISPLAY_NAME[gesture]}", font=chinese_font(36), fill=(0, 0, 0))
    draw.text((30, 78), "操作方式：以石頭前後搖晃，定住後出拳", font=chinese_font(25), fill=(0, 0, 0))
    status = throw_status or "請以石頭搖晃準備出拳"
    draw.text((30, 126), f"出拳狀態：{status}", font=chinese_font(30), fill=(0, 0, 0))
    if result:
        player, computer, verdict = result
        draw.text((30, 170), f"你：{DISPLAY_NAME[player]}　電腦：{DISPLAY_NAME[computer]}",
                  font=chinese_font(30), fill=(0, 0, 0))
        draw.text((30, 218), f"結果：{verdict}", font=chinese_font(32), fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


@dataclass
class ThrowDetector:
    """Recognize a rock shake followed by a stable hand as one completed throw."""

    positions: Deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=45))
    stable_frames: int = 0
    armed: bool = False
    round_started: bool = False

    def update(self, hand, gesture: str) -> Optional[str]:
        """Return ``shake`` when a new round starts and ``throw`` when it ends."""
        # Once the player has started a rock shake, keep tracking even when
        # the final throw changes to scissors or paper.
        if not self.armed and not self.round_started and gesture != "rock":
            self.positions.clear()
            self.stable_frames = 0
            self.round_started = False
            return None

        pts = hand
        # Average palm points rather than use a fingertip, which moves in a fist.
        p = np.mean([(pts[i].x, pts[i].y, pts[i].z) for i in (0, 5, 9, 13, 17)], axis=0)
        self.positions.append(tuple(float(v) for v in p))
        if len(self.positions) < 10:
            return None

        recent = np.asarray(list(self.positions)[-8:])
        motion = float(np.max(np.linalg.norm(np.diff(recent[:, :2], axis=0), axis=1)))
        # A held hand naturally jitters in camera coordinates.  This generous
        # tolerance treats that small movement as a completed, stable throw.
        if motion < 0.020:
            self.stable_frames += 1
        else:
            self.stable_frames = 0

        history = np.asarray(self.positions)
        # One clear direction reversal is sufficient for a natural shake; two
        # was too strict and caused valid throws in the test video to be lost.
        x_steps = np.diff(history[:, 0])
        signs = [1 if step > 0.006 else -1 if step < -0.006 else 0 for step in x_steps]
        compact = [sign for sign in signs if sign]
        reversals = sum(a != b for a, b in zip(compact, compact[1:]))
        # Clear the previous result as soon as the player starts the familiar
        # rock-shaking preparation, rather than waiting until the throw is armed.
        shake_started = gesture == "rock" and not self.round_started and float(np.ptp(history[:, 0])) > 0.025
        if shake_started:
            self.round_started = True
        # The final hand pose may no longer be a rock.  A sufficiently broad
        # shake is enough to arm the pending throw regardless of that change.
        if self.round_started and (float(np.ptp(history[:, 0])) > 0.075 or reversals >= 1):
            if not self.armed:
                self.armed = True

        # Never lock an ambiguous pose.  Require a fresh run of stable frames
        # after the recognizer returns one of the three valid RPS gestures.
        if gesture == "unknown":
            self.stable_frames = 0
            return None

        if self.armed and self.stable_frames >= 10:
            self.armed = False
            self.round_started = False
            self.positions.clear()
            self.stable_frames = 0
            return "throw"
        return "shake" if shake_started else None


def run(source: str | int, headless: bool, expected: Optional[list[str]]) -> int:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    images = load_choice_images()
    detector = ThrowDetector()
    observed: list[str] = []
    last_result: Optional[tuple[str, str, str]] = None
    throw_status: Optional[str] = None
    model_path = ROOT / "gesture_recognizer.task"
    if not model_path.exists():
        raise FileNotFoundError("Missing gesture_recognizer.task next to rps_game.py")
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    with mp.tasks.vision.GestureRecognizer.create_from_options(options) as tracker:
        frame_number = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_number * 1000 / fps)
            result = tracker.recognize_for_video(mp_image, timestamp_ms)
            frame_number += 1
            gesture = gesture_from_result(result)
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand:
                draw_landmarks(frame, hand)
                event = detector.update(hand, gesture)
                if event == "shake" and last_result:
                    # A new rock shake explicitly starts a fresh round.
                    last_result = None
                    throw_status = "準備出拳"
                elif event == "shake":
                    throw_status = "準備出拳"
                elif event == "throw":
                    throw_status = "手勢鎖定"
                    computer = random.choice(CHOICES)
                    last_result = (gesture, computer, outcome(gesture, computer))
                    observed.append(gesture)

            draw_status_panel(frame, gesture, last_result, throw_status)
            if last_result:
                _, computer, _ = last_result
                paste_choice(frame, images[computer])
            if not headless:
                cv2.imshow("Rock Paper Scissors", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    cap.release()
    cv2.destroyAllWindows()
    if expected is not None:
        print("Detected throws:", " -> ".join(observed) or "(none)")
        print("Expected throws:", " -> ".join(expected))
        if observed != expected:
            print("TEST FAILED: detected sequence differs from the supplied video.")
            return 1
        print("TEST PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MediaPipe rock-paper-scissors")
    parser.add_argument("--video", help="Video path; omit for the webcam")
    parser.add_argument("--headless", action="store_true", help="Do not open a preview window")
    parser.add_argument("--verify-test", action="store_true", help="Assert the known sequence in test.mov")
    args = parser.parse_args()
    source: str | int = args.video if args.video else 0
    expected = ["rock", "scissors", "paper", "scissors", "paper"] if args.verify_test else None
    return run(source, args.headless, expected)


if __name__ == "__main__":
    raise SystemExit(main())
