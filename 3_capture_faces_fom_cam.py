import dlib
import numpy as np
import cv2
import os
import shutil
import time

# Dlib 
detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = "models/model_faces_from_camera/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0         # TO count faces
        self.ss_cnt = 0                     # To count screenshots
        self.current_frame_faces_cnt = 0    # To count current frame's faces

        self.save_flag = 1                  # To save
        self.press_n_flag = 0               # to create folder before saving

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

    
    def pre_work_mkdir(self):
        
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

   
    def pre_work_del_old_face_folders(self):

        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera+folders_rd[i])
        if os.path.isfile("models/features_all.csv"):
            os.remove("models/features_all.csv")

    
    def check_existing_faces_cnt(self):
        if os.listdir("models/model_faces_from_camera/"):
            
            person_list = os.listdir("models/model_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)

        
        else:
            self.existing_faces_cnt = 0


    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now


    def draw_note(self, img_rd):

        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

# Main code block

    def process(self, stream):
        # create directory
        self.pre_work_mkdir()


        if os.path.isdir(self.path_photos_from_camera):
            self.pre_work_del_old_face_folders()


        self.check_existing_faces_cnt()

        while stream.isOpened():
            flag, img_rd = stream.read()        
            kk = cv2.waitKey(1)
            faces = detector(img_rd, 0)         


            if kk == ord('n'):
                self.existing_faces_cnt += 1
                current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                os.makedirs(current_face_dir)
                print('\n')
                print("Create folders: ", current_face_dir)

                self.ss_cnt = 0                 
                self.press_n_flag = 1           

           
            if len(faces) != 0:
               
                for k, d in enumerate(faces):

                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height/2)
                    ww = int(width/2)


                    if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                        cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if kk == ord('s'):
                            print("Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)


                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                    if save_flag:

                        if kk == ord('s'):

                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for ii in range(height*2):
                                    for jj in range(width*2):
                                        img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                                cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                print(" Save into：", str(current_face_dir) + "/img_face_" + str(self.ss_cnt) + ".jpg")
                            else:
                                print(" Please press 'N' and press 'S'")

            self.current_frame_faces_cnt = len(faces)


            self.draw_note(img_rd)


            if kk == ord('q'):
                break

            # 11. Update FPS
            self.update_fps()

            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)

    def run(self):
        cap = cv2.VideoCapture(0)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
