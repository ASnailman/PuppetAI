import cv2 as cv
import pyautogui
import mediapipe as mp

def hand_tracking():
    # initialize mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils # Initialize MediaPipe Drawing module (for debugging - draw hand landmarks)
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands # Initialize MediaPipe Hands module
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) # Initialize MediaPipe Hands with confidence thresholds

    # vars
    cap = cv.VideoCapture(0) # '0' is the default camera
    prev_y = 0
    tolerance = 0.02
    difference = 0.0
    screen_width, screen_height = pyautogui.size() # get screen size

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
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                    pointer = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pointer_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    ring_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    pinkie = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    pinkie_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    # move mouse with pointer
                    if (pointer_knuckle.y > pointer.y and 
                        middle_knuckle.y < middle.y and 
                        thumb.x > thumb_knuckle.x and 
                        ring_knuckle.y < ring.y and 
                        pinkie_knuckle.y < pinkie.y):
                        pyautogui.moveTo(int(pointer.x * screen_width), int(pointer.y * screen_height))
                    
                    # scroll with pointer and middle
                    if (pointer_knuckle.y > pointer.y and 
                        middle_knuckle.y > middle.y and 
                        thumb.x > thumb_knuckle.x and 
                        ring_knuckle.y < ring.y and 
                        pinkie_knuckle.y < pinkie.y):
                        if abs(difference) > tolerance:
                            if pointer.y and middle.y > prev_y:
                                scroll_amt = int(250 * difference)
                                pyautogui.scroll(scroll_amt)
                            elif pointer.y < prev_y:
                                scroll_amt = int(250 * difference)
                                pyautogui.scroll(scroll_amt)
                        if pointer.y and middle.y < 0.55 and pointer.y and middle.y > 0.45:
                            prev_y = pointer.y
                        difference = float(pointer.y - prev_y)
                                
            cv.imshow('Live Feed', frame) # display frame

            if cv.waitKey(1) & 0xFF == ord('q'): # exit when q key is pressed
                break
    finally:
        cap.release() # release camera
        cv.destroyAllWindows() # Close all OpenCV windows

def main():
    hand_tracking()

if __name__ == "__main__":
    main()