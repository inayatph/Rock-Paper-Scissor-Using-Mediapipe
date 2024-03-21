import mediapipe as mp
import cv2

# Initialize MediaPipe Hands module
mphands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands object
Hand = mphands.Hands(max_num_hands=1)

# Open camera feed
video = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    success, image = video.read()
    image = cv2.flip(image, 1)
    
    # If frame is not successfully read, skip iteration
    if not success:
        print('Ignore empty camera frame')
        continue
    
    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    result = Hand.process(img_rgb)
    
    # Define fingertip IDs
    tipids = [4, 8, 12, 16, 20]
    
    # List to store landmark positions
    lmlist = []
    
    # Draw rectangle for displaying gestures
    cv2.rectangle(image, (20, 350), (150, 400), (0, 255, 255), cv2.FILLED)

    # Check if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get landmark coordinates
                cx = lm.x
                cy = lm.y
                lmlist.append([id, cx, cy])

                # Detect gestures
                if len(lmlist) != 0 and len(lmlist) == 21:
                    fingerlist = []

                    # Detect fingers (thumb and other fingers)
                    if lmlist[20][1] > lmlist[12][1]:
                        if lmlist[4][1] > lmlist[3][1]:
                            fingerlist.append(0)
                        else:
                            fingerlist.append(1)
                    else:
                        if lmlist[4][1] < lmlist[3][1]:
                            fingerlist.append(0)
                        else:
                            fingerlist.append(1)

                    for i in range(1, 5):
                        if lmlist[tipids[i]][2] > lmlist[tipids[i] - 2][2]:
                            fingerlist.append(0)
                        else:
                            fingerlist.append(1)

                    # Count raised fingers
                    fingercount = fingerlist.count(1)
                    print("Finger count:", fingercount)

                    # Determine gesture based on finger count
                    gesture = ""
                    if fingercount == 0:
                        gesture = "Rock"
                    elif fingercount == 2:
                        gesture = "Scissor"
                    elif fingercount == 5:
                        gesture = "Paper"
                  

                    # Display gesture
                    cv2.putText(image, gesture, (30, 390), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3))

    # Display the annotated image
    cv2.imshow('Hand Gesture Recognition', image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()