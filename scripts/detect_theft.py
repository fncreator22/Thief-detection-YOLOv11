import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train12/weights/best.pt")



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  
    if not ret:
        break

   
    results = model.predict(frame, conf=0.5)


    for r in results:
        for box in r.boxes:
          
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = box.conf[0]                     
            cls = int(box.cls[0])                  
            label = model.names[cls]               

   
            if label == 'alert':
                color = (0, 0, 255) 
            elif label == 'normal':
                color = (0, 255, 0) 
            else:
                color = (255, 255, 0)  


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    cv2.imshow("Theft Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
