

#Face recognition for multiple face

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time


detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor('models/model_dlib/shape_predictor_68_face_landmarks.dat')


face_reco_model = dlib.face_recognition_model_v1("models/model_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # For FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        # cnt for frame
        self.frame_cnt = 0


        self.features_known_list = []

        self.name_known_list = []


        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []


        self.last_frame_names_list = []
        self.current_frame_face_name_list = []


        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0


        self.current_frame_face_X_e_distance_list = []


        self.current_frame_face_position_list = []

        self.current_frame_face_features_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0


    def get_face_database(self):
        if os.path.exists("models/features_all.csv"):
            path_features_known_csv = "models/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.features_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_" + str(i + 1))
            print("Faces in Database：", len(self.features_known_list))
            return 1
        else:
            print('##### Warning #####', '\n')
            print("'features_all.csv' not found!")
            print(
                "Please run '3_capture_faces_fom_cam.py' and '4_feature_extraction_as_csv.py' before '5_recognise_face_from_cam.py'",
                '\n')
            print('##### End Warning #####')
            return 0


    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now


    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist


    def centroid_tracker(self):
        for i in range(len(self.current_frame_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_centroid_list[i], self.last_frame_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]


    def draw_note(self, img_rd):
        
        cv2.putText(img_rd, "Face recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            cv2.putText(img_rd, "Face " + str(i + 1), tuple(
                [int(self.current_frame_centroid_list[i][0]), int(self.current_frame_centroid_list[i][1])]), self.font,
                        0.8, (255, 190, 0),
                        1,
                        cv2.LINE_AA)


    def process(self, stream):

        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                print(">>> Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)


                faces = detector(img_rd, 0)
                if self.current_frame_face_name_list == ['Person_2', 'Person_2']:
                    break

                # Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)
                # Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]
                # update frame centroid list
                self.last_frame_centroid_list = self.current_frame_centroid_list
                self.current_frame_centroid_list = []
                print("   >>> current_frame_face_cnt: ", self.current_frame_face_cnt)

                # 2.1. if cnt not changes
                if self.current_frame_face_cnt == self.last_frame_face_cnt:
                    print("   >>> scene 1: no faces cnt changes in this frame!!!")
                    self.current_frame_face_position_list = []
                    if self.current_frame_face_cnt != 0:
                        # 2.1.1 Get ROI positions
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])


                            height = (d.bottom() - d.top())
                            width = (d.right() - d.left())
                            hh = int(height / 2)
                            ww = int(width / 2)
                            cv2.rectangle(img_rd,
                                          tuple([d.left() - ww, d.top() - hh]),
                                          tuple([d.right() + ww, d.bottom() + hh]),
                                          (255, 255, 255), 2)

                    # multi-faces in current frames, use centroid tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 write names under ROI
                        cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                    self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                    cv2.LINE_AA)

                # 2.2 if cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    print("   >>> scene 2:  Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []

                    # 2.2.1 face cnt decrease: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        print("   >>> scene 2.1  No guy in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                        self.current_frame_face_features_list = []

                    # 2.2.2 face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        print("   >>> scene 2.2 Do face recognition for people detected in this frame")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_features_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")


                        for k in range(len(faces)):
                            print("      >>> For face " + str(k+1) + " in current frame:")
                            self.current_frame_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []


                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))


                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.features_known_list)):

                                if str(self.features_known_list[i][0]) != '0.0':
                                    print("            >>> with person", str(i + 1), "the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_features_list[k],
                                        self.features_known_list[i])
                                    print(e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:

                                    self.current_frame_face_X_e_distance_list.append(999999999)


                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.name_known_list[similar_person_num]
                                print("            >>> recognition result for face " + str(k+1) +": "+ self.name_known_list[similar_person_num])
                            else:
                                print("            >>> recognition result for face " + str(k + 1) + ": " + "unknown")

                self.draw_note(img_rd)


                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)
                print(">>> Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(0)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
