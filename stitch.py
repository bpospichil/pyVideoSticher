import cv
import cv2
import numpy as np
from matplotlib import pyplot as plt

def showImage(img1, 
    pts=None,
    title="Image"):

    # # Show me the tracked points!
    if ( len(img1.shape) == 3 ):
        composite_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    else:
        composite_image = np.zeros((img1.shape[0], img1.shape[1]), np.uint8)

    composite_image[:,0:img1.shape[1]] += img1

    if ( pts != None ):
        for pt in pts:
            cv2.circle(composite_image, (pt[0], pt[1]), 5, 255)
    
    cv2.imshow(title, composite_image)


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cam0 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    cam0mtx = np.asarray(cv.Load("cam0.xml", cv.CreateMemStorage(), 'camera_matrix'))
    cam0dist = np.asarray(cv.Load("cam0.xml", cv.CreateMemStorage(), 'distortion_coefficients'))
    cam2mtx = np.asarray(cv.Load("cam2.xml", cv.CreateMemStorage(), 'camera_matrix'))
    cam2dist = np.asarray(cv.Load("cam2.xml", cv.CreateMemStorage(), 'distortion_coefficients'))

    while (True):

        ret1, frame1 = cam0.read()
        h, w = frame1.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam0mtx,cam0dist,(w,h),1,(w,h))
        # undistort
        frame1 = cv2.undistort(frame1, cam0mtx, cam0dist, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        frame1 = frame1[y:y+h, x:x+w]

        ret2, frame2 = cam2.read()
        h, w = frame2.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam2mtx,cam2dist,(w,h),1,(w,h))
        # undistort
        frame2 = cv2.undistort(frame2, cam2mtx, cam2dist, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        frame2 = frame2[y:y+h, x:x+w]

        # Use the SIFT feature detector
        detector = cv2.SIFT()

        # Convert and blur frame1 image
        frame1_blur = cv2.GaussianBlur(cv2.cvtColor(frame1.copy(), cv2.COLOR_BGR2GRAY), (5,5), 0)
        
        # Find key points in base image for motion estimation
        f1_features, f1_descs = detector.detectAndCompute(frame1_blur, None) 

        # Create new key point list
        f1_key_points = []
        for kp in f1_features:
            f1_key_points.append((int(kp.pt[0]),int(kp.pt[1])))
        showImage(frame1_blur, f1_key_points, 'frame1')


        # Convert and blur frame2 image
        frame2_blur = cv2.GaussianBlur(cv2.cvtColor(frame2.copy(), cv2.COLOR_BGR2GRAY), (5,5), 0)
        
        # Find key points in base image for motion estimation
        f2_features, f2_descs = detector.detectAndCompute(frame2_blur, None) 

        # Create new key point list
        f2_key_points = []
        for kp in f2_features:
            f2_key_points.append((int(kp.pt[0]),int(kp.pt[1])))
        showImage(frame2_blur, f2_key_points, 'frame2')

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(f1_descs,f2_descs, k=2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.5*m[1].distance]

        img3 = drawMatches(frame1_blur,f1_features,frame2_blur,f2_features,matches)
        plt.imshow(img3),plt.show()



#        cv2.imshow("cam0", frame1_blur)
    #    cv2.imshow("cam0", frame1)
    #    cv2.imshow("cam2", frame2)
        if cv2.waitKey() == 1048689:
            break

