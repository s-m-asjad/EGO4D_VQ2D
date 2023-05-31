import cv2
import sys
import numpy as np

try:
    folder = sys.argv[1]
    predicted_response_track = sys.argv[2]
    ground_truth_response_track = sys.argv[3]

except:
    # USE THIS TO HELP YOU WITH THE FORMATTING OF INPUT ARGUMENT
    predicted_response_track = [[[{'bboxes': [{'fno': 172, 'x1': 1514, 'x2': 1919, 'y1': 777, 'y2': 1080}], 'score': 1.0}]]] 
    ground_truth_response_track = [{'bboxes': [{'fno': 116, 'x1': 472, 'x2': 763, 'y1': 959, 'y2': 1080}, {'fno': 117, 'x1': 509, 'x2': 837, 'y1': 920, 'y2': 1078}, {'fno': 118, 'x1': 478, 'x2': 830, 'y1': 897, 'y2': 1080}, {'fno': 119, 'x1': 371, 'x2': 732, 'y1': 882, 'y2': 1080}], 'score': None}]
    folder = "experiment_tyre_base"


cap = cv2.VideoCapture('/home/asjad.s/EGO4D/experiments/'+folder+'/visual_queries_logs/example_00000_sw.mp4')
print("Video Loaded")

predicted_response_track = predicted_response_track[0][0][0]["bboxes"]
ground_truth_response_track = ground_truth_response_track[0]["bboxes"]

frames = []

ret = True
filename = '/home/asjad.s/EGO4D/experiments/'+folder+'/visual_queries_logs/visualizer.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, 30, (1088, 1920) )
while(cap.isOpened()):
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        #frames.append(img)
        vidout=img
        out.write(vidout)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        break



cap.release()
out.release()
cv2.destroyAllWindows()



