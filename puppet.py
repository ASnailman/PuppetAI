import cv2 as cv
import pyautogui
import mediapipe as mp
import tkinter as tk
import threading
from puppet_functions.point_overlay import PointOverlay

def hand_tracking(overlay, screen_width, screen_height):
    # initialize mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils # Initialize MediaPipe Drawing module (for debugging - draw hand landmarks)
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands # Initialize MediaPipe Hands module
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) # Initialize MediaPipe Hands with confidence thresholds

    # vars
    cap = cv.VideoCapture(0) # '0' is the default camera
    prev_y = None
    
    try:
        while True:
            # read returns ret (return) and frame, ret is boolean val that indicates if frame was captured
            ret, frame = cap.read()
            if not ret:
                break

            # flips frame horizontally to face user
            frame = cv.flip(frame, 1) # 1 flips horizontally along y-axis

            # Convert the BGR image to RGB, opencv uses BGR by default, MediaPipe uses RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Mark the image as not writeable to improve performance
            rgb_frame.flags.writeable = False
            # Process the frame with MediaPipe Hands
            results = hands.process(frame)
            rgb_frame.flags.writeable = True

            # scroll functionality
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                    pointer = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pointer_y = pointer.y

                    tolerance = 0.01
                    if prev_y is not None:
                        if pointer_y > prev_y - tolerance:
                            pyautogui.scroll(100)
                        elif pointer_y < prev_y + tolerance:
                            pyautogui.scroll(-100)
                    prev_y = pointer_y

                    fingertip_x = int(pointer.x * screen_width)
                    fingertip_y = int(pointer.y * screen_height)

                    overlay.root.after(0, overlay.update_dot, fingertip_x, fingertip_y)
                                
            cv.imshow('Live Feed', frame) # display frame

            if cv.waitKey(1) & 0xFF == ord('q'): # exit when q key is pressed
                overlay.root.destroy()
                break
    finally:
        cap.release() # release camera
        cv.destroyAllWindows() # Close all OpenCV windows

def main():
    screen_width, screen_height = pyautogui.size() # get screen size

    overlay = PointOverlay() # initialize overlay window

    thread = threading.Thread(target=hand_tracking, args=(overlay, screen_width, screen_height), daemon=True) # avoid running hand tracking in main thread, daemon ensures thread stops when main program exits
    thread.start()

    overlay.run()

if __name__ == "__main__":
    main()