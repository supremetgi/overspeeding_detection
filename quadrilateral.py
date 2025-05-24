import cv2

# Store clicked points
points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

# Load the video
cap = cv2.VideoCapture("output_trimmed.mp4")  # Replace with your video file path

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw all selected points
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Draw a quadrilateral if 4 points selected
    if len(points) == 4:
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC key to exit
        break
    elif key == ord('c'):  # Press 'c' to clear points
        points = []

cap.release()
cv2.destroyAllWindows()
