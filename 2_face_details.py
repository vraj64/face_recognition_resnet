# Real-time face descriptor compute

import dlib         
import cv2          
import time


# 1. Dlib
detector = dlib.get_frontal_face_detector()

# 2. Dlib landmark
predictor = dlib.shape_predictor('models/model_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 128D
face_reco_model = dlib.face_recognition_model_v1("models/model_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Descriptor:
    def __init__(self):
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()

    def process(self, stream):
        while stream.isOpened():
            flag, img_rd = stream.read()
            k = cv2.waitKey(1)

            faces = detector(img_rd, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX

            
            if len(faces) != 0:
                for face in faces:
                    face_shape = predictor(img_rd, face)
                    face_desc = face_reco_model.compute_face_descriptor(img_rd, face_shape)

            # BGR format:
            cv2.putText(img_rd, "Face Details", (20, 60), font, 1, (51, 87, 255), 2, cv2.LINE_AA) 
            cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (1, 254, 1), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "No of. Faces: " + str(len(faces)), (20, 140), font, 0.75, (1, 254, 1), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "S: Save this face", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            # press 'q' to break
            if k == ord('q'):
                break

            self.update_fps()

            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)


def main():
    Face_Descriptor_con = Face_Descriptor()
    Face_Descriptor_con.run()


if __name__ == '__main__':
    main()
