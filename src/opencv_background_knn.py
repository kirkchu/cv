import cv2

bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (400, 225))
    frame = cv2.flip(frame, 1)
    gray = bs.apply(frame)
    mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.hconcat([frame, mask])
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break
