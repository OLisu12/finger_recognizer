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

    step = 1

    number1 = None
    operator = None
    number2 = None
    result_value = None

    finger_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(
            rgb_frame
        )

        finger_count = 0

        if (
            results.multi_hand_landmarks
            and results.multi_handedness
        ):

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = (
                    handedness
                    .classification[0]
                    .label
                )

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
            f"Fingers: {finger_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        if step == 1:
            message = (
                "Show first number + SPACE"
            )

        elif step == 2:
            message = (
                "1:+  2:-  3:*  4:/"
            )

        elif step == 3:
            message = (
                "Show second number + SPACE"
            )

        else:
            message = (
                f"Result: {result_value}"
            )

        cv2.putText(
            frame,
            message,
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"{number1} {operator} {number2}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE: Confirm   R: Reset   Q: Quit",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Finger Calculator",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

        if key == ord("r"):
            step = 1

            number1 = None
            operator = None
            number2 = None
            result_value = None

        if key == ord(" "):

            if step == 1:
                number1 = finger_count

                step = 2

            elif step == 2:
                operators = {
                    1: "+",
                    2: "-",
                    3: "*",
                    4: "/"
                }

                if finger_count in operators:
                    operator = operators[
                        finger_count
                    ]

                    step = 3

            elif step == 3:
                number2 = finger_count

                if operator == "+":
                    result_value = (
                        number1
                        + number2
                    )

                elif operator == "-":
                    result_value = (
                        number1
                        - number2
                    )

                elif operator == "*":
                    result_value = (
                        number1
                        * number2
                    )

                elif operator == "/":

                    if number2 == 0:
                        result_value = "Error"

                    else:
                        result_value = (
                            number1
                            / number2
                        )

                step = 4

    hands.close()

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


