import cv2
import numpy as np

AGE_BUCKETS = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)',
    '(25-32)', '(38-43)', '(48-53)', '(60-100)'
]

GENDER_LIST = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load networks
face_net = cv2.dnn.readNet(
    "opencv_face_detector_uint8.pb",
    "opencv_face_detector.pbtxt"
)

age_net = cv2.dnn.readNet(
    "age_net.caffemodel",
    "age_deploy.prototxt"
)

gender_net = cv2.dnn.readNet(
    "gender_net.caffemodel",
    "gender_deploy.prototxt"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        [104, 117, 123], swapRB=False
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.85:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                MODEL_MEAN_VALUES, swapRB=False
            )

            # Gender
            gender_net.setInput(face_blob)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            # Age
            age_net.setInput(face_blob)
            age = AGE_BUCKETS[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

    cv2.imshow("Gender & Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
