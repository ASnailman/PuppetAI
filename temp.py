import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe Drawing module (for debugging - draw hand landmarks)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV for accessing the camera
cap = cv2.VideoCapture(0)  # '0' is the default camera

# Define variables to track finger movement
previous_y = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for natural viewing
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks (for debugging purposes)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the y-coordinate of the tip of the index finger (landmark 8)
                finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                # Compare with the previous frame to determine the movement direction
                if previous_y is not None:
                    if finger_tip_y < previous_y - 0.01:  # Adjust sensitivity as needed
                        pyautogui.scroll(-100)  # Scroll down
                    elif finger_tip_y > previous_y + 0.01:
                        pyautogui.scroll(100)  # Scroll up

                # Update the previous y-coordinate
                previous_y = finger_tip_y

        # Display the frame
        cv2.imshow('Live Feed', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()