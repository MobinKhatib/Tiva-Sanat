from ultralytics import YOLO
import cv2
import math
import time
import datetime

# Get the current timestamp for output names
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Define the file name with the timestamp
file_name = f'output_{timestamp}.jpg'
classnames = ['car', 'plate']
charclassnames = ['0','9','b','d','ein','ein','g','gh','h','n','s','1','malul','n','s','sad','t','ta','v','y','2'
                  ,'3','4','5','6','7','8']
source = "assets/video.mp4"
#load YOLOv8 model
model_object = YOLO("weights/best.pt")
model_char = YOLO("weights/yolov8n_char_new.pt")


cap = cv2.VideoCapture(source)
# Define the output video properties
output_videoname = f'output_{timestamp}.mp4'
output_imagename = f'output_{timestamp}.jpg'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames > 1:
    video_writer = cv2.VideoWriter('output/' + output_videoname, fourcc, fps, (frame_width, frame_height))
    #do inference for video
    while cap.isOpened():
        success, img = cap.read()
        if success:
            #detect objects with yolov8s model
            tick = time.time()
            output = model_object(img, show=False, conf=0.7, stream=True)
            #extract bounding box and class names
            for i in output:
                bbox = i.boxes
                for box in bbox:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img,(x1, y1), (x2, y2), (208, 224, 64), 3)
                    confs = math.ceil((box.conf[0]*100))/100
                    cls_names = int(box.cls[0])
                    if cls_names == 1:
                        cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)
                    elif cls_names == 0:
                        cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)
                    
                    #check plate to recognize characters with yolov8n model
                    if cls_names == 1:
                        char_display = []
                        #crop plate from frame
                        plate_img = img[ y1:y2, x1:x2]
                        #detect characters of plate with yolov8n model
                        #plate_output = model_char(plate_img, conf=0.2)
                        conf_threshold = 0.3  # Start with an initial value
                        while conf_threshold >= 0.05:  # Minimum confidence threshold
                            plate_output = model_char(plate_img, conf=conf_threshold)
                            if plate_output[0].boxes:  # Check if any detections are found
                                break
                            conf_threshold -= 0.05  # Lower the confidence incrementally
                        tock_2 = time.time()
                        elapsed_time_2 = tock_2 - tick
                        
                        #extract bounding box and class names
                        bbox = plate_output[0].boxes.xyxy
                        cls = plate_output[0].boxes.cls
                        #make a dict and sort it from left to right to show the correct characters of plate
                        keys = cls.cpu().numpy().astype(int)
                        values = bbox[:, 0].cpu().numpy().astype(int)
                        dictionary = list(zip(keys, values))
                        sorted_list = sorted(dictionary, key=lambda x: x[1])
                        #convert all characters to a string
                        for i in sorted_list:
                            char_class = i[0]
                            char_display.append(charclassnames[char_class])
                        char_result ='Plate: ' + (''.join(char_display))
                        fps_text_2 = "FPS: {:.2f}".format(1/elapsed_time_2)
                        text_size, _ = cv2.getTextSize(fps_text_2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
                        #just show the correct characters in output
                        if len(char_display) == 8:
                            cv2.line(img, (max(40, x1 - 25 ), max(40, y1 - 10)), (x2 + 25 ,y1 - 10), (208, 224, 64), 20,lineType=cv2.LINE_AA)
                            cv2.putText(img, char_result , (max(40, x1 - 15), max(40, y1 - 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255),thickness=1, lineType=cv2.LINE_AA)
                tock = time.time()
                elapsed_time = tock - tick
                fps_text = "FPS: {:.2f}".format(1/elapsed_time)
                text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                fps_text_loc = (frame_width - text_size[0] - 10, text_size[1] + 10)
                cv2.putText(img, fps_text , fps_text_loc, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(10, 50, 255),thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('detection', img)
            video_writer.write(img)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

                        
else: #do inference for image
    output = model_object(source, show=False, conf=0.75)
    img = cv2.imread(source)
    #extract bounding box and class names
    for i in output:
        bbox = i.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1, y1), (x2, y2), (208, 224, 64), 1)
            confs = math.ceil((box.conf[0]*100))/100
            cls_names = int(box.cls[0])
            if cls_names == 1:
                cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)
            elif cls_names == 0:
                cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(0, 20, 255),thickness=1, lineType=cv2.LINE_AA)

            #check plate to recognize characters with yolov8n model
            if cls_names == 1:
                char_display = []
                #crop plate from frame
                plate_img = img[y1:y2, x1:x2]
                #detect characters of plate with yolov8n model
                #plate_output = model_char(plate_img, conf=0.2)
                #print("plate output\n", plate_output)
                conf_threshold = 0.9
                best_char_display = None
                closest_distance = float('inf')  # Initialize closest distance to a very high value

                while conf_threshold >= 0.1:  # Adjust the confidence threshold
                    plate_output = model_char(plate_img, conf=conf_threshold)
                    if plate_output and plate_output[0].boxes:  # Check if any detections are found
                        # Extract bounding box and class names
                        bbox = plate_output[0].boxes.xyxy
                        cls = plate_output[0].boxes.cls
                        keys = cls.cpu().numpy().astype(int)
                        values = bbox[:, 0].cpu().numpy().astype(int)
                        dictionary = list(zip(keys, values))
                        sorted_list = sorted(dictionary, key=lambda x: x[1])  # Sort by x-coordinate (left to right)

                        # Convert all characters to a string
                        char_display = [charclassnames[i[0]] for i in sorted_list]

                        # Calculate the distance from 8 characters
                        current_distance = abs(len(char_display) - 8)

                        # Update the best_char_display if this detection is closer to 8 characters
                        if current_distance < closest_distance:
                            closest_distance = current_distance
                            best_char_display = char_display

                        print(f"Detections found with {len(char_display)} characters. Distance from 8: {current_distance}.")
                    else:
                        print("No detections found at this confidence threshold.")

                    conf_threshold -= 0.05  # Faster decrement

                # Further processing and displaying the result
                if best_char_display:
                    char_result = ''.join(best_char_display)
                    print(f"Best detection result: {char_result} with {len(best_char_display)} characters (closest to 8).")
                    cv2.line(img, (max(40, x1 - 25), max(40, y1 - 10)), (x2 + 25, y1 - 10), (0, 0, 0), 20, lineType=cv2.LINE_AA)
                    cv2.putText(img, char_result, (max(40, x1 - 15), max(40, y1 - 5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (10, 50, 255), 1, cv2.LINE_AA)
                else:
                    print("No valid detections found.")

                cv2.imshow('detection', img)
                cv2.imwrite('output/' + output_imagename, img)
    # exit if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()