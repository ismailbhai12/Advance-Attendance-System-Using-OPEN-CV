# pylint: disable=no-member

import cv2 # it handles the computer vision function
from simple_facerec import SimpleFacerec  # for better face recognition

from datetime import datetime  # used to record the timestamp
import csv  # it handles the csv files

# Encoding all the faces which is in our folder 'images'
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# It will load our default camera
web_cam = cv2.VideoCapture(0)

# initiallize / creating the 'attendance.csv' file if it's not exist
# it will store as -> 'Name' & 'Timestamp'
csv_file = 'Attendance.csv'

current_date = datetime.now().strftime('%d/%m/%Y')

with open(csv_file , 'w' , newline= '') as file :
    writer = csv.writer(file)
    writer.writerow([f"                                                      ATTENDANCE   ({current_date})"])
    writer.writerow(['Name ' , ' Time Of Attendance'])

# this set will store the name of the student who have already been present 
# to mainly avoid duplicate entries
marked_student = set()

# Initialize a flag to track when no face is detected
no_face_detected = True

while True:
    ret, frame = web_cam.read()

    # Detect the faces
    face_location, face_names = sfr.detect_known_faces(frame)

    if face_names:
        # If faces are detected, process them
        for face_loc, name in zip(face_location, face_names):
            y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            
            # Most of the time 'Font Hershey' should be used otherwise it's showing error
            cv2.putText(frame, name, (x1, y1-15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # marking present for the students who have not been marked present earlier
            if name not in marked_student : 
                
                time_of_attendance = datetime.now().strftime('%H:%M:%S')
                print(f"{name} : Present at {time_of_attendance}")
                
                # also writing their name and datetime in the csv file
                with open(csv_file , 'a' , newline= '') as file :
                    writer = csv.writer(file)
                    writer.writerow([name , datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                
                # adding the name to avoid duplicate entries   
                marked_student.add(name)
                
        # Reset the no-face-detected flag since a face is detected
        no_face_detected = False

    else:
        # If no faces are detected and it hasn't been recently printed
        if not no_face_detected:
            print("Next student")
            
            no_face_detected = True  # Set flag to prevent repeated prints
    
    # Display the frame
    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

web_cam.release()
cv2.destroyAllWindows()
