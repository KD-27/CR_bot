#Creating an environment to load all the libraries

cd ~/CR_bot
python3 -m venv venv
source venv/bin/activate

#Installing Yolov8(ultralytics), opencv-python and all other dependencies
resource - https://www.geeksforgeeks.org/computer-vision/object-detection-with-yolo-and-opencv/

pip install ultralytics opencv-python
--> Check Inside Lib_test/yolo_cv_test folder

