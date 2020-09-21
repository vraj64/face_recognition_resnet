import cv2

cap = cv2.VideoCapture(0)

print(cap.isOpened())

# cap.read()


while cap.isOpened():
    ret_flag, img_camera = cap.read()

    print("height: ", img_camera.shape[0])
    print("width:  ", img_camera.shape[1])
    print('\n')
    #The default 480*640 will be displayed!
    cv2.imshow("camera", img_camera)

  
    k = cv2.waitKey(1)

    # press 't' to take an image
    if k == ord('t'):
        cv2.imwrite("test.jpg", img_camera)

    # press 'q' to quit
    if k == ord('q'):
        break


cap.release()


cv2.destroyAllWindows()
