import cv2
import numpy as np

# Define the four corner points of the quadrilateral
points = [(846, 335), (1412, 335), (1502, 531), (712, 531)]
pts = np.array(points, np.int32).reshape((-1, 1, 2))

# Open your input video
cap = cv2.VideoCapture("output_trimmed.mp4")  # Replace with your video path

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter('output_with_quad.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw outline
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Optional: Transparent fill inside the polygon
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Write the frame to output
    out.write(frame)

    # Optional: Show the frame while processing
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Output saved as 'output_with_quad.mp4'")
