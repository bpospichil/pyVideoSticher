import cv
import cv2
import math
import numpy as np
from numpy import linalg
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

def findDimensions(image, homography):

    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
        hp_arr = np.array(hp, np.float32)
        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]
        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]
        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]
        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)


if __name__ == '__main__':
    cam0 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)
    cam0mtx = np.asarray(cv.Load("cam0.xml", cv.CreateMemStorage(), 'camera_matrix'))
    cam0dist = np.asarray(cv.Load("cam0.xml", cv.CreateMemStorage(), 'distortion_coefficients'))
    cam2mtx = np.asarray(cv.Load("cam2.xml", cv.CreateMemStorage(), 'camera_matrix'))
    cam2dist = np.asarray(cv.Load("cam2.xml", cv.CreateMemStorage(), 'distortion_coefficients'))

    while (True):

        ret1, frame1 = cam0.read()
        #h, w = frame1.shape[:2]
        #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam0mtx,cam0dist,(w,h),1,(w,h))
        # undistort
        #frame1 = cv2.undistort(frame1, cam0mtx, cam0dist, None, newcameramtx)
        # crop the image
        #x,y,w,h = roi
        #frame1 = frame1[y:y+h, x:x+w]

        ret2, frame2 = cam2.read()
        #h, w = frame2.shape[:2]
        #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cam2mtx,cam2dist,(w,h),1,(w,h))
        # undistort
        #frame2 = cv2.undistort(frame2, cam2mtx, cam2dist, None, newcameramtx)
        # crop the image
        #x,y,w,h = roi
        #frame2 = frame2[y:y+h, x:x+w]

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
        matches = bf.knnMatch(f2_descs,trainDescriptors=f1_descs, k=2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.5*m[1].distance]

        # Create Homography matrix
        kp1 = [f1_features[match.trainIdx] for match in matches]
        kp2 = [f2_features[match.queryIdx] for match in matches]
        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

        # Normalize and invert
        H = H / H[2,2]
        H_inv = linalg.inv(H)

        (min_x, min_y, max_x, max_y) = findDimensions(frame2_blur, H_inv)
        
        # Adjust max_x and max_y frame1 size
        max_x = max(max_x, frame1_blur.shape[1])
        max_y = max(max_y, frame1_blur.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if (min_x < 0):
            move_h[0,2] += -min_x
            max_x += -min_x

        if (min_y < 0):
            move_h[1,2] += -min_y
            max_y += -min_y

        mod_inv_h = move_h * H_inv

        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))

        # Warp the new image given the homography from the old image
        frame1_warp = cv2.warpPerspective(frame1, move_h, (img_w, img_h))
        frame2_warp = cv2.warpPerspective(frame2, mod_inv_h, (img_w, img_h))

        cv2.imshow("cam0", frame1_warp)
        cv2.imshow("cam2", frame2_warp)


        # Put the base image on an enlarged palette
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        # Create a mask from the warped image for constructing masked composite
        (ret,data_map) = cv2.threshold(cv2.cvtColor(frame2_warp, cv2.COLOR_BGR2GRAY), 
            0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, frame1_warp,
                                    mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)

        # Now add the warped image
        final_img = cv2.add(enlarged_base_img, frame2_warp, dtype=cv2.CV_8U)

        # Crop off the black edges
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = 0
        best_rect = (0,0,0,0)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            deltaHeight = h-y
            deltaWidth = w-x
            area = deltaHeight * deltaWidth
            if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                max_area = area
                best_rect = (x,y,w,h)

        if (max_area > 0):
            final_img = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                                  best_rect[0]:best_rect[0]+best_rect[2]]

        cv2.imshow("teste", final_img)
#        img3 = drawMatches(frame1_blur,f1_features,frame2_blur,f2_features,matches)
#        plt.imshow(img3),plt.show()

        





#        cv2.imshow("cam0", frame1_blur)
    #    cv2.imshow("cam0", frame1)
    #    cv2.imshow("cam2", frame2)
        if cv2.waitKey() == 1048689:
            break

