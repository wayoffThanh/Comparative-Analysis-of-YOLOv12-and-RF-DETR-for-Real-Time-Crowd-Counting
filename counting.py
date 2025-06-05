import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("6622416238885.mp4")
assert cap.isOpened(), "Error reading video file"


region_points = [[8, 215], [10, 896], [1588, 900], [1588, 215]]

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="best.pt",

)


while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)



    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()