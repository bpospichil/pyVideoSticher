import cv
import cv2
import numpy


def showImage(img1, 
    pts=None,
    title="Image"):

    # # Show me the tracked points!
    if ( len(img1.shape) == 3 ):
        composite_image = numpy.zeros((img1.shape[0], img1.shape[1], 3), numpy.uint8)
    else:
        composite_image = numpy.zeros((img1.shape[0], img1.shape[1]), numpy.uint8)

    composite_image[:,0:img1.shape[1]] += img1

    if ( pts != None ):
        for pt in pts:
            cv2.circle(composite_image, (pt[0], pt[1]), 5, 255)
    
    cv2.imshow(title, composite_image)


if __name__ == '__main__':
    cam0 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    cam0mtx = numpy.asarray(cv.Load("cam0.xml", cv.CreateMemStorage(), 'camera_matrix'))
    cam0dist = numpy.asarray(cv.Load("cam0.xml", cv.CreateMemStorage(), 'distortion_coefficients'))
    cam2mtx = numpy.asarray(cv.Load("cam2.xml", cv.CreateMemStorage(), 'camera_matrix'))
    cam2dist = numpy.asarray(cv.Load("cam2.xml", cv.CreateMemStorage(), 'distortion_coefficients'))

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

        
        #Create the feature matcher
        
        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, 
            trees = 5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})

        matches = matcher.knnMatch(f2_descs, trainDescriptors=f1_descs, k=2)

        print "\t Match Count: ", len(matches)

#        cv2.imshow("cam0", frame1_blur)
    #    cv2.imshow("cam0", frame1)
    #    cv2.imshow("cam2", frame2)
        if cv2.waitKey() == 1048689:
            break

