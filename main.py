import cv2
import mediapipe as mp




def count_fingers(hand_landmarks, hand_label):
   landmarks = hand_landmarks.landmark
   finger_count = 0


   finger_pairs = [
       (8, 6),
       (12, 10),
       (16, 14),
       (20, 18)
   ]


   for tip, middle_joint in finger_pairs:
       if landmarks[tip].y < landmarks[middle_joint].y:
           finger_count += 1


   if hand_label == "Right":
       if landmarks[4].x < landmarks[3].x:
           finger_count += 1


   else:
       if landmarks[4].x > landmarks[3].x:
           finger_count += 1


   return finger_count




def main():
   cap = cv2.VideoCapture(0)


   if not cap.isOpened():
       print("웹캠을 열 수 없습니다.")
       return


   mp_hands = mp.solutions.hands
   mp_draw = mp.solutions.drawing_utils


   hands = mp_hands.Hands(
       static_image_mode=False,
       max_num_hands=1,
       min_detection_confidence=0.7,
       min_tracking_confidence=0.7
   )


   while True:
       ret, frame = cap.read()


       if not ret:
           break


       frame = cv2.flip(frame, 1)


       rgb_frame = cv2.cvtColor(
           frame,
           cv2.COLOR_BGR2RGB
       )


       result = hands.process(rgb_frame)


       finger_count = 0
       hand_label_text = "No Hand"


       if result.multi_hand_landmarks and result.multi_handedness:
           for hand_landmarks, handedness in zip(
               result.multi_hand_landmarks,
               result.multi_handedness
           ):
               hand_label = (
                   handedness
                   .classification[0]
                   .label
               )


               hand_label_text = hand_label


               finger_count = count_fingers(
                   hand_landmarks,
                   hand_label
               )


               mp_draw.draw_landmarks(
                   frame,
                   hand_landmarks,
                   mp_hands.HAND_CONNECTIONS
               )


       cv2.putText(
           frame,
           f"Hand: {hand_label_text}",
           (30, 50),
           cv2.FONT_HERSHEY_SIMPLEX,
           1,
           (255, 255, 255),
           2
       )


       cv2.putText(
           frame,
           f"Fingers: {finger_count}",
           (30, 110),
           cv2.FONT_HERSHEY_SIMPLEX,
           1.5,
           (255, 255, 255),
           3
       )


       cv2.putText(
           frame,
           "Press Q to quit",
           (30, 160),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.8,
           (255, 255, 255),
           2
       )


       cv2.imshow(
           "Finger Counter",
           frame
       )


       if cv2.waitKey(1) & 0xFF == ord("q"):
           break


   hands.close()
   cap.release()
   cv2.destroyAllWindows()




if __name__ == "__main__":
   main()


