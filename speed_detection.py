import cv2

# Load video
cap = cv2.VideoCapture("output_with_quad.mp4")

# Get video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter("output_cv_tracking_speed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Background subtractor
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)


# Define distance between entry and exit lines (meters)
distance_meters = 7

# Define entry and exit Y-coordinate positions (top to bottom in image)
entry_line = 531  # Y-coordinate where car enters
exit_line = 335   # Y-coordinate where car exits

# Dictionary to track vehicle entry frame number and their centroid
vehicle_tracks = {}
vehicle_counter = 0
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Convert to grayscale & blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Background subtraction
    fgmask = fgbg.apply(gray)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # Ignore small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2  # Y-center is what we care about (vehicles move top-down)

            matched_id = None
            for vid, data in vehicle_tracks.items():
                _, prev_cy, _, entry_frame, exited = data
                if abs(cy - prev_cy) < 50 and not exited:
                    matched_id = vid
                    break

            if matched_id is None:
                # New vehicle detected
                vehicle_counter += 1
                vehicle_tracks[vehicle_counter] = [cx, cy, frame_number, frame_number, False]
                matched_id = vehicle_counter
            else:
                # Update existing vehicle
                vehicle_tracks[matched_id][0] = cx
                vehicle_tracks[matched_id][1] = cy
                vehicle_tracks[matched_id][3] = frame_number

            entry_frame = vehicle_tracks[matched_id][2]
            exited = vehicle_tracks[matched_id][4]

            if not exited and cy < exit_line:
                exit_frame = frame_number
                time_seconds = (exit_frame - entry_frame) / fps
                speed_kmph = (distance_meters / time_seconds) * 3.6 if time_seconds > 0 else 0

                # Mark as exited
                vehicle_tracks[matched_id][4] = True

                # Draw bounding box and speed
                if speed_kmph > 5:
                    color = (0, 140, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green

                cv2.putText(frame, f"{int(speed_kmph)} km/h", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            elif not exited:
                # Vehicle still between entry and exit
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(frame)
    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()  