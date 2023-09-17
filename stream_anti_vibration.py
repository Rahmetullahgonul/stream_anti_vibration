# camera stream captured and stabilized using a kalman filter 
# https://nghiaho.com/uploads/videostabKalman.cpp
import cv2
import numpy as np

# constants
HORIZONTAL_BORDER_CROP=20

# video capture
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("error: video file not found or could not be opened")
    exit()

# output video writer
fourcc=cv2.VideoWriter_fourcc(*'XVID')
frame_rate=int(cap.get(cv2.CAP_PROP_FPS))
frame_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
outputVideo=cv2.VideoWriter('compare.avi',fourcc,frame_rate,frame_size,True)

# variables
prev=None
prev_grey=None
prev_corner=None
cur=None
cur_grey=None
cur_corner=None
last_T=None
x,y,a=0,0,0

# kalman filter variables
X=np.zeros(3,dtype=np.float32)
X_=np.zeros(3)
P=np.eye(3,dtype=np.float32)
P_=np.eye(3)
K=np.zeros(3,dtype=np.float32)
Z=np.zeros(3)
pstd = 4e-3
cstd = 0.25
Q=np.array([pstd,pstd,pstd])
R=np.array([cstd,cstd,cstd])

kalman_K=1
max_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret,cur=cap.read()
    if not ret:
        break

    cur_grey=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    if prev is None:
        prev=cur.copy()
        prev_grey=cur_grey.copy()
        continue

    # vector from prev  to cur
    prev_corner=cv2.goodFeaturesToTrack(prev_grey,200,0.01,30)
    cur_corner,status,_=cv2.calcOpticalFlowPyrLK(prev_grey,cur_grey,prev_corner,None)

    # weed out bad matches
    prev_corner=prev_corner[status==1]
    cur_corner=cur_corner[status==1]

    # translation + rotation only
    T=cv2.estimateAffine2D(prev_corner,cur_corner,False)[0]
    if T is None:
        T=np.eye(2,3,dtype=np.float32)


    # decompose T
    dx=T[0,2]
    dy=T[1,2]
    da=np.arctan2(T[1,0],T[0,0])

    x+=dx
    y+=dy
    a+=da

    # kalman filter 
    if kalman_K==1:
        X=np.array([0,0,0])
        P=np.eye(3)
    else:
        # time update (prediction)
        kalman_K=kalman_K+1

        # measurement update(correction)
        K=P_ / (P_+R)
        X=X_+K*(Z+X_)
        P=(np.eye(3)-K)*P_
    
    diff_x=X[0]-x
    diff_y=X[1]-y
    diff_a=X[2]-a

    dx +=diff_x
    dy +=diff_y
    da +=diff_a

    # update transform matrix
    T[0,0]=np.cos(da)
    T[0,1]=-np.sin(da)
    T[1,0]=np.sin(da)
    T[1,1]=np.cos(da)
    T[0,2]=dx
    T[1,2]=dy

    # apply the transformation to the current frame
    cur2=cv2.warpAffine(prev,T,(prev.shape[1],prev.shape[0]),flags=cv2.INTER_LINEAR)

    # crop the black borders
    cur2=cur2[HORIZONTAL_BORDER_CROP:-HORIZONTAL_BORDER_CROP,:]

    # resize 
    cur2=cv2.resize(cur2,(prev.shape[1],prev.shape[0]))

    # stack the original and stabilized frames side by side
    canvas=np.zeros((cur.shape[0],cur.shape[1]*2+10,cur.shape[2]),dtype=np.uint8)
    canvas[:,:cur.shape[1]]=prev
    canvas[:,cur.shape[1]+10:]=cur2

    # write the frame to the output video
    outputVideo.write(canvas)

    # display the frame
    cv2.imshow("Before and after",canvas)

    # update variables
    prev=cur.copy()
    prev_grey=cur_grey.copy()

    print(f"Frame: {K}/{max_frames} - Good optical flow: {prev_corner.shape[0]}")
    K += 1

    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

cap.release()
outputVideo.release()
cv2.destroyAllWindows()