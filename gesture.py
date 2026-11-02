import cv2
import mediapipe
import time
import serial
import math
from collections import deque, Counter

# =========================================================
# ARDUINO SERIAL CONNECTION
# =========================================================
ser = serial.Serial('COM4', 9600, timeout=1)

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================================================
# MEDIAPIPE
# =========================================================
medhands = mediapipe.solutions.hands

hands = medhands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

draw = mediapipe.solutions.drawing_utils

# =========================================================
# GESTURE SETTINGS
# =========================================================

# Number of frames required to confirm a gesture
STABLE_FRAMES = 8

# Stores recent finger counts
gesture_history = deque(maxlen=STABLE_FRAMES)

# Current gesture which has already triggered
active_gesture = None

# Time protection
last_send_time = 0
COMMAND_DELAY = 0.5

# =========================================================
# DISTANCE FUNCTION
# =========================================================

def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# =========================================================
# ANGLE FUNCTION
# =========================================================

def calculate_angle(a, b, c):

    ba = (
        a[0] - b[0],
        a[1] - b[1]
    )

    bc = (
        c[0] - b[0],
        c[1] - b[1]
    )

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine_angle = max(
        -1,
        min(1, cosine_angle)
    )

    angle = math.degrees(
        math.acos(cosine_angle)
    )

    return angle


# =========================================================
# FINGER COUNT
# =========================================================

def count_fingers(lmlist):

    # Wrist
    wrist = (
        lmlist[0][1],
        lmlist[0][2]
    )

    # Finger landmark IDs
    fingers = [

        # Index
        (5, 6, 7, 8),

        # Middle
        (9, 10, 11, 12),

        # Ring
        (13, 14, 15, 16),

        # Little
        (17, 18, 19, 20)
    ]

    count = 0

    # Palm size for normalization
    palm_size = distance(
        lmlist[0][1],
        lmlist[0][2],
        lmlist[9][1],
        lmlist[9][2]
    )

    if palm_size == 0:
        return 0

    for mcp_id, pip_id, dip_id, tip_id in fingers:

        mcp = (
            lmlist[mcp_id][1],
            lmlist[mcp_id][2]
        )

        pip = (
            lmlist[pip_id][1],
            lmlist[pip_id][2]
        )

        dip = (
            lmlist[dip_id][1],
            lmlist[dip_id][2]
        )

        tip = (
            lmlist[tip_id][1],
            lmlist[tip_id][2]
        )

        # ---------------------------------------------
        # METHOD 1: Finger angle
        # ---------------------------------------------

        angle = calculate_angle(
            mcp,
            pip,
            dip
        )

        # ---------------------------------------------
        # METHOD 2: Distance from wrist
        # ---------------------------------------------

        tip_distance = distance(
            wrist[0],
            wrist[1],
            tip[0],
            tip[1]
        )

        pip_distance = distance(
            wrist[0],
            wrist[1],
            pip[0],
            pip[1]
        )

        # Normalize distances
        normalized_tip = tip_distance / palm_size
        normalized_pip = pip_distance / palm_size

        # ---------------------------------------------
        # Finger considered OPEN when:
        #
        # 1. Finger angle is reasonably straight
        # AND
        # 2. Fingertip is farther from wrist
        # ---------------------------------------------

        if (
            angle > 155
            and normalized_tip > normalized_pip * 1.12
        ):
            count += 1

    return count


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, img = cap.read()

    if not success:
        continue

    # Mirror camera
    img = cv2.flip(img, 1)

    # Convert to RGB
    imgrgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe processing
    res = hands.process(imgrgb)

    lmlist = []

    # =====================================================
    # DRAW DETECTION BOX
    # =====================================================

    cv2.rectangle(
        img,
        (20, 350),
        (110, 440),
        (0, 255, 204),
        cv2.FILLED
    )

    cv2.rectangle(
        img,
        (20, 350),
        (110, 440),
        (0, 0, 0),
        5
    )

    # =====================================================
    # HAND DETECTED
    # =====================================================

    if res.multi_hand_landmarks:

        handlms = res.multi_hand_landmarks[0]

        h, w, c = img.shape

        # ---------------------------------------------
        # Store landmarks
        # ---------------------------------------------

        for id, lm in enumerate(handlms.landmark):

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmlist.append([
                id,
                cx,
                cy
            ])

        # =================================================
        # FINGER COUNT
        # =================================================

        if len(lmlist) == 21:

            fingercount = count_fingers(lmlist)

            # Safety limit
            if fingercount > 4:
                fingercount = 4

            # ---------------------------------------------
            # Add count to history
            # ---------------------------------------------

            gesture_history.append(fingercount)

            # ---------------------------------------------
            # Check whether all recent frames agree
            # ---------------------------------------------

            if len(gesture_history) == STABLE_FRAMES:

                counter = Counter(
                    gesture_history
                )

                stable_gesture, frequency = counter.most_common(1)[0]

                # Require strong agreement
                if frequency >= 6:

                    # =====================================
                    # NEUTRAL / RESET
                    # =====================================

                    if stable_gesture == 0:

                        active_gesture = None

                    # =====================================
                    # GESTURES 1 - 4
                    # =====================================

                    elif stable_gesture in [1, 2, 3, 4]:

                        current_time = time.time()

                        # Only send if this is a NEW gesture
                        if (
                            stable_gesture != active_gesture
                            and
                            current_time - last_send_time
                            > COMMAND_DELAY
                        ):

                            command = f"*{stable_gesture}"

                            ser.write(
                                command.encode()
                            )

                            print(
                                f"GESTURE {stable_gesture} "
                                f"-> {command}"
                            )

                            # Remember this gesture
                            active_gesture = stable_gesture

                            last_send_time = current_time

            # =================================================
            # DISPLAY
            # =================================================

            display_gesture = fingercount

            cv2.putText(
                img,
                str(display_gesture),
                (42, 425),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 0, 0),
                5
            )

        # =================================================
        # DRAW LANDMARKS
        # =================================================

        draw.draw_landmarks(
            img,
            handlms,
            medhands.HAND_CONNECTIONS,

            draw.DrawingSpec(
                color=(0, 255, 204),
                thickness=2,
                circle_radius=3
            ),

            draw.DrawingSpec(
                color=(0, 0, 0),
                thickness=2,
                circle_radius=2
            )
        )

    # =====================================================
    # NO HAND DETECTED
    # =====================================================

    else:

        # Clear history when hand disappears
        gesture_history.clear()

        # Do NOT reset active_gesture here.
        #
        # This is important.
        #
        # If the camera temporarily loses the hand,
        # showing the same gesture again should NOT
        # accidentally toggle the bulb.
        #
        # The user must show 0 / neutral to reset.

    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(
        img,
        "GESTURE: 1-4",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # =====================================================
    # DISPLAY WINDOW
    # =====================================================

    cv2.imshow(
        "hand gestures",
        img
    )

    # =====================================================
    # EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================================================
# CLOSE EVERYTHING
# =========================================================

cap.release()

cv2.destroyAllWindows()

ser.close()