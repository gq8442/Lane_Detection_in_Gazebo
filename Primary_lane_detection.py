#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 01:22:00 2020

@author: kush
"""
import sys
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import cv2, cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image

class Follower:
    def __init__(self):
        '''
        INITIALISE FUNCTION TO INITIALISE THE FOLLOWER CLASS
        Returns
        -------
        CAMERA FROM THE PUBLISHED ROS TOPIC
        '''
        # The CvBridge is an object that converts between OpenCV Images and ROS Image messages.
        self.bridge = cv_bridge.CvBridge()
        # subscribing to the ROS topic /camera_front/image_raw
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.image_callback)	

    def image_callback(self, msg, sigma=0.23):
        '''
        Parameters
        ----------
        msg : cv2 Image
        
        Returns
        -------
        THERE ARE TWO LOCAL FUNCTIONS reg_of_interest AND draw_lines
            - reg_of_interest FUNCTION RETURNS THE MASKED IMAGE OF THE INTERESTED REGION IN WHICH THE LANES ARE PRESENT.
            - draw_lines FUNCTION TAKES THE ORIGINAL IMAGE AND ADDS WEIGHTED LINE VECTORS ON TO IT
            - THE MAIN FUNCTION IS USED TO DO THE HOUGH LINES TRANSFORMATION AND SHOW THE CV2 WINDOW               
        '''
        def reg_of_interest(img, vertices):
            # TAKES THE CANNY IMAGE AND CONVERTS ALL THE PIXEL VALUES TO ZERO
            mask = np.zeros_like(img)
            match_mask_color = 255
            # MASK OR FILL THE COLOR OUTSIDE OF VERTICES EQUAL TO match_mask_color
            cv2.fillPoly(mask, vertices, match_mask_color)
            # OVERLAP ORIGINAL IMAGE AND MAKSED REGION
            masked_image = cv2.bitwise_and(img, mask)
            # RETURN masked_image TO THE FUNCTION
            return masked_image

        def draw_lines(img, line_vector):
            # NUMPY COPY OF THE ORIGINAL IMAGE FOR DETECTING THE LINE VECTORS
            img = np.copy(img)
            # BLANK IMAGE WITH THE SAME WIDTH AND HEIGHT AS OF THE IMAGE COMING FROM INIT FUNCTION
            blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for line in line_vector:
                # TUPLES OF THE X AND Y COORDINATES COMING FROM THE LINE LIST
                for x1,y1,x2,y2 in line:
                    # ADDING OR CREATING THE CV2 LINE ON THE BLANK IMAGE
                    cv2.line(blank_img, (x1,y1), (x2,y2), (0,0,255), thickness=3)
            #OVERLAPPING BALNK IMAGE WITH LINES WITH THE ORIGINAL IMAGE
            img = cv2.addWeighted(img, 0.6, blank_img, 1, 0.0)
            # RETURNING THE IMAGE WITH LINES TO THE FUNCTION
            return img
        
        # To convert a ROS image message into an cv::Mat
        # more information at - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        image = self.bridge.imgmsg_to_cv2(msg)
        """
        When the image file is read with the cv2 function above, the order of colors is 
        BGR (blue, green, red). To change this order from BGR to RGB below is used.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # first value of the .shape method for the image is height and second value is width
        height = image.shape[0]
        width = image.shape[1]
        # define the region of interest vertices manually here
        regio_of_int_vertices = [(213,655),(403,340),(618,340),(712,505),(797,655)]
        # conert the RGB image to gray image using cv2.COLOR_RGB2GRAY
        blur_img = cv2.medianBlur(image, 7)
        gray_image = cv2.cvtColor(blur_img, cv2.COLOR_RGB2GRAY)
        v = np.median(gray_image)
        lowlim = int(max(0, (1.0 - sigma) * v))
        upplim = int(min(255, (1.0 + sigma) * v))
        # applying canny method on the gray image for edge detection
        canny_image = cv2.Canny(gray_image, lowlim, upplim)
        # crop the canny image using the reg_of_interest function
        cropped_image = reg_of_interest(canny_image, np.array([regio_of_int_vertices], np.int32))
        # detect edges in the cropeed canny image using the Hough line transform
        lines = cv2.HoughLinesP(cropped_image, rho=3, theta=np.pi/60, threshold=100,
                          lines=np.array([]), minLineLength=80, maxLineGap=50)
        # execute the draw_lines function so with the original image and detected lines arguments
        image_lines = draw_lines(image, lines)
        # show the original image with the line
        cv2.imshow("window",image)
        cv2.waitKey(2)    


if __name__ == '__main__':
    # EXECUTE THE THE FOLLOWER CLASS WHICH WILL INITIALISE THE follower NODE
    rospy.init_node('follower')
    follower = Follower()
    rospy.spin()
