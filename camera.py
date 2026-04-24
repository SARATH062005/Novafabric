import cv2

# Try 0 first. If not working, try 1, 2, or 3
camera_index = 2  

cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Cannot open USB camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read frame")
        break

    cv2.imshow("USB Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
