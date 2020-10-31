# Strain-Analysis-Based-On-Eye-Blinking
System that monitors and alerts a stressed person based on eye aspect ratio
## Description
A Computer Vision system that alerts the person if eyes are getting strained. This System uses the integrated webcam to capture the eyes of the person and counts the number of times a person blinks. If blink count deviates from the average value, then the system alerts the person by playing an audio message along with a popup message on the screen.
## Prerequisites
1. Install python 3.6
2. Install OpenCV 
3. Install imutils
4. Install Google text to speech 
5. Install playsound
6. Install Dlib
7. Download dlib shape predictor
## Run the Application
To run the code, execute the following command
<b> python app_eye3.py --shape-predictor shape_predictor_68_face_landmarks.dat </b>
## Output
![](https://github.com/Meghana-U/Strain-Analysis-Based-On-Eye-Blinking/blob/main/output.PNG)
