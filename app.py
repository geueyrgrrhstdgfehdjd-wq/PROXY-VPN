import threading
import time
import os

from flask import Flask, render_template_string, jsonify

try:
    import numpy as np
    import cv2
    import mss
    import pyautogui
    AIMBOT_AVAILABLE = True
except Exception:
    AIMBOT_AVAILABLE = False

app = Flask(__name__)
aimbot_active = False
thread = None

# ========== ปรับสีตามจริง ==========
# สีเอาต์ไลน์ศัตรู (แดง/ม่วง)
LOWER_RED = np.array([140, 80, 80])
UPPER_RED = np.array([180, 255, 255])
LOWER_PURPLE = np.array([130, 50, 80])
UPPER_PURPLE = np.array([160, 255, 255])

# สีเอาต์ไลน์เพื่อน (เขียว/ฟ้า) ต้องตัดทิ้ง
LOWER_FRIEND_GREEN = np.array([35, 50, 50])
UPPER_FRIEND_GREEN = np.array([85, 255, 255])
LOWER_FRIEND_BLUE = np.array([100, 80, 80])
UPPER_FRIEND_BLUE = np.array([130, 255, 255])

# สัดส่วนหัวจากขอบบนของบอดี้
HEAD_RATIO = 0.22
HEAD_MAX_RATIO = 0.35
MIN_CONTOUR_AREA = 80
BODY_ASPECT_MIN = 1.2
BODY_ASPECT_MAX = 4.5
CLICK_DELAY = 0.18
LOCK_FRAMES_REQUIRED = 2   # ต้องล็อกหัวนิ่ง 2 เฟรมก่อนยิง

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def preprocess_mask(frame):
    """สร้างมาสก์เฉพาะศัตรู ตัดเพื่อนออก"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    enemy_red = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    enemy_purple = cv2.inRange(hsv, LOWER_PURPLE, UPPER_PURPLE)
    enemy_mask = cv2.bitwise_or(enemy_red, enemy_purple)

    friend_green = cv2.inRange(hsv, LOWER_FRIEND_GREEN, UPPER_FRIEND_GREEN)
    friend_blue = cv2.inRange(hsv, LOWER_FRIEND_BLUE, UPPER_FRIEND_BLUE)
    friend_mask = cv2.bitwise_or(friend_green, friend_blue)

    # ลบเพื่อนออกจากแมสก์ศัตรู
    enemy_mask = cv2.bitwise_and(enemy_mask, cv2.bitwise_not(friend_mask))

    kernel = np.ones((5, 5), np.uint8)
    enemy_mask = cv2.morphologyEx(enemy_mask, cv2.MORPH_OPEN, kernel)
    enemy_mask = cv2.dilate(enemy_mask, kernel, iterations=2)
    return enemy_mask

def is_valid_body(contour):
    """เช็กว่าคอนทัวร์เป็นคน ไม่ใช่ฉาก/UI"""
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return False
    aspect = h / w
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return False
    if aspect < BODY_ASPECT_MIN or aspect > BODY_ASPECT_MAX:
        return False
    # คนต้องสูงมากกว่ากว้างอย่างชัดเจน
    if h < 40:
        return False
    return True

def find_enemy_head(frame, enemy_mask):
    """คืนหัวศัตรูเฉพาะตัวจริง และต้องเป็นพิกเซลของศัตรู"""
    contours, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_head = None
    best_dist = float("inf")

    # จอปัจจุบัน
    screen_h, screen_w = frame.shape[:2]
    center_x, center_y = screen_w // 2, screen_h // 2

    for c in contours:
        if not is_valid_body(c):
            continue

        x, y, w, h = cv2.boundingRect(c)
        # หัวอยู่ช่วงบน 22-35% ของความสูง
        head_y_low = y + int(h * HEAD_RATIO)
        head_y_high = y + int(h * HEAD_MAX_RATIO)

        # หาจุดหัวที่อยู่ในแมสก์ศัตรู (ไม่ใช่ช่องว่าง)
        head_candidates = []
        for hy in range(head_y_low, min(head_y_high, y + h - 1)):
            for hx in range(x, x + w):
                if enemy_mask[hy, hx] > 0:
                    head_candidates.append((hx, hy))

        if not head_candidates:
            continue

        # เลือกจุดหัวบนสุดใกล้กลางตัว
        head_x = sum(p[0] for p in head_candidates) // len(head_candidates)
        head_y = min(p[1] for p in head_candidates)

        # ระยะจากหัวถึงกลางจอ เอาตัวที่ใกล้สุด
        dist = np.sqrt((head_x - center_x) ** 2 + (head_y - center_y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_head = (head_x, head_y, (x, y, w, h))

    return best_head

def aim_and_shoot():
    global aimbot_active
    if not AIMBOT_AVAILABLE:
        aimbot_active = False
        return

    pyautogui.FAILSAFE = True
    lock_counter = 0
    last_target = None

    while aimbot_active:
        try:
            frame = capture_screen()
            enemy_mask = preprocess_mask(frame)
            target = find_enemy_head(frame, enemy_mask)

            if target:
                head_x, head_y, bbox = target
                screen_w, screen_h = pyautogui.size()
                center_x, center_y = screen_w // 2, screen_h // 2

                # ตรวจว่ายังอยู่ในแมสก์ศัตรูจริง (กันจอสั่น/เป้าหมายหาย)
                if head_x < enemy_mask.shape[1] and head_y < enemy_mask.shape[0]:
                    if enemy_mask[head_y, head_x] == 0:
                        target = None

            if target:
                head_x, head_y, bbox = target
                center_x, center_y = pyautogui.size()[0] // 2, pyautogui.size()[1] // 2

                if abs(head_x - center_x) > 2 or abs(head_y - center_y) > 2:
                    pyautogui.moveTo(head_x, head_y, duration=0.04)
                    lock_counter = 0
                else:
                    lock_counter += 1
                    if lock_counter >= LOCK_FRAMES_REQUIRED:
                        # ก่อนคลิก ตรวจซ้ำว่าจุดนั้นเป็นศัตรู ไม่ใช่พื้น/กำแพง
                        frame_now = capture_screen()
                        mask_now = preprocess_mask(frame_now)
                        if head_y < mask_now.shape[0] and head_x < mask_now.shape[1]:
                            if mask_now[head_y, head_x] > 0:
                                pyautogui.click()
                                time.sleep(CLICK_DELAY)
                                lock_counter = 0
            else:
                lock_counter = 0

            time.sleep(0.01)
        except Exception as e:
            print(f"[AIMBOT] error: {e}")
            time.sleep(0.5)

# ================== เว็บคุม ==================
@app.route("/")
def index():
    status = "กำลังทำงาน" if aimbot_active else "ปิดอยู่"
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Free Fire Aimbot - Head Only</title>
        <style>
            body { font-family: 'Courier New', monospace; background: #111; color: #0f0; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            button { background: #0f0; color: #000; border: none; padding: 15px 30px; font-size: 20px; cursor: pointer; margin: 10px; font-weight: bold; }
            button:disabled { background: #333; color: #666; cursor: not-allowed; }
            p { font-size: 18px; }
        </style>
    </head>
    <body>
        <h1>Bluez GFT Head Aimbot</h1>
        <p>สถานะ: {{ status }}</p>
        <button onclick="activate()">เปิดใช้งาน</button>
        <button onclick="deactivate()">ปิดใช้งาน</button>
        <p id="msg"></p>
        <script>
            async function activate() {
                const res = await fetch('/activate', { method: 'POST' });
                const data = await res.json();
                document.getElementById('msg').textContent = data.message;
                location.reload();
            }
            async function deactivate() {
                const res = await fetch('/deactivate', { method: 'POST' });
                const data = await res.json();
                document.getElementById('msg').textContent = data.message;
                location.reload();
            }
        </script>
    </body>
    </html>
    """, status=status)

@app.route("/activate", methods=["POST"])
def activate():
    global aimbot_active, thread
    if not aimbot_active:
        aimbot_active = True
        thread = threading.Thread(target=aim_and_shoot, daemon=True)
        thread.start()
        message = "เปิดใช้งานแล้ว"
    else:
        message = "เปิดใช้งานอยู่แล้ว"
    return jsonify({"success": True, "message": message})

@app.route("/deactivate", methods=["POST"])
def deactivate():
    global aimbot_active
    aimbot_active = False
    return jsonify({"success": True, "message": "ปิดใช้งานแล้ว"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting web on port {port}, aimbot_available={AIMBOT_AVAILABLE}")
    app.run(host="0.0.0.0", port=port)
