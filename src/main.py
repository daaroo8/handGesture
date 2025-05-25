import cv2
import mediapipe as mp
import pygame as pg
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pg.mixer.init()


def load_gestures_images():
    gestures_name = [
        "givemefive", "okey", "blm", "rock", "yey", "fucku", "perfect", "peace",
        "surf", "bad", "punch", "promise", "italian", "tiny", "pistol", "uwu",
        "looser", "friki", "vulcano", "no_gesture"
    ]

    gestures_imgs = {}

    for gesture in gestures_name:
        img_path = f"./imgs/{gesture}.png"

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[ERROR] No se pudo cargar: {img_path}")
            continue

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        gestures_imgs[gesture] = img

    return gestures_imgs


def overlay_image(background, overlay, pos=(0, 0)):
    if overlay is None:
        return background

    h, w = background.shape[:2]
    overlay_resized = cv2.resize(overlay, (w // 5, h // 5), interpolation=cv2.INTER_AREA)

    h_overlay, w_overlay = overlay_resized.shape[:2]
    x, y = pos

    x = min(x, w - w_overlay)
    y = min(y, h - h_overlay)

    roi = background[y:y + h_overlay, x:x + w_overlay]

    if overlay_resized.shape[2] == 4:
 
        overlay_rgb = overlay_resized[:, :, :3].astype(np.float32)
        alpha = overlay_resized[:, :, 3].astype(np.float32) / 255.0
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        roi = roi.astype(np.float32)

        blended = (1.0 - alpha) * roi + alpha * overlay_rgb
        background[y:y + h_overlay, x:x + w_overlay] = blended.astype(np.uint8)
    else:
  
        background[y:y + h_overlay, x:x + w_overlay] = overlay_resized

    return background


def is_finger_down(landmarks, finger_tip, finger_pip):

  if finger_tip == 4:
    return landmarks[finger_tip].x > landmarks[finger_pip].x
  
  else:
    return landmarks[finger_tip].y > landmarks[finger_pip].y
  

def is_hand_open(landmarks, tips, pips):
  return not (is_finger_down(landmarks, tips[0], pips[0]) and \
     is_finger_down(landmarks, tips[1], pips[1]) and \
     is_finger_down(landmarks, tips[2], pips[2]) and \
     is_finger_down(landmarks, tips[3], pips[3]) and \
     is_finger_down(landmarks, tips[4], pips[4]))
  
def hand_turned(landmarks, mcps, threshold=0.15):
    hand = landmarks.landmark
    return (hand[mcps[4]].y - hand[mcps[1]].y) > threshold

def hand_flipped(hand, mcps, threshold=0.15):
    return (hand[mcps[1]].x - hand[mcps[4]].x) > threshold

def are_touching(hand, tip1, tip2):
    return abs(hand[tip1].x - hand[tip2].x) < 0.05 and abs(hand[tip1].y - hand[tip2].y) < 0.05

def okey_gesture(landmarks, mcps, tips, pips):
    hand = landmarks.landmark

    thumb_up = hand[tips[0]].y < hand[mcps[0]].y  

    fingers_bent = all(
        hand[tips[i]].x > hand[pips[i]].x
        for i in range(1, 4)
    )

    return thumb_up and fingers_bent and hand_turned(landmarks, mcps)

def bad_gesture(landmarks, tips, pips, mcps):
    hand = landmarks.landmark

    thumb_down = hand[tips[0]].y > hand[pips[0]].y

    fingers_bent = all(
        hand[tips[i]].x > hand[pips[i]].x
        for i in range(1, 4)
    )

    return thumb_down and fingers_bent and hand[mcps[1]].y > hand[mcps[4]].y

def rock_gesture(landmarks, tips, pips):
    hand = landmarks.landmark

    return is_finger_down(hand, tips[0], pips[0]) and \
           not is_finger_down(hand, tips[1], pips[1]) and \
           is_finger_down(hand, tips[2], pips[2]) and \
           is_finger_down(hand, tips[3], pips[3]) and \
           not is_finger_down(hand, tips[4], pips[4])

def yey_gesture(landmarks, tips, pips):
    hand = landmarks.landmark

    return not is_finger_down(hand, tips[0], pips[0]) and \
           not is_finger_down(hand, tips[1], pips[1]) and \
           is_finger_down(hand, tips[2], pips[2]) and \
           is_finger_down(hand, tips[3], pips[3]) and \
           not is_finger_down(hand, tips[4], pips[4])

def fuck_you(landmarks, tips, pips, mcps):
  hand = landmarks.landmark

  return hand_flipped(hand, mcps) and \
         is_finger_down(hand, tips[0], pips[0]) and \
         is_finger_down(hand, tips[1], pips[1]) and \
         not is_finger_down(hand, tips[2], pips[2]) and \
         is_finger_down(hand, tips[3], pips[3]) and \
         is_finger_down(hand, tips[4], pips[4])

def perfect_gesture(landmarks, tips, pips):

  hand = landmarks.landmark

  return are_touching(hand, tips[0], tips[1]) and \
         not is_finger_down(hand, tips[2], pips[2]) and \
         not is_finger_down(hand, tips[3], pips[3]) and \
         not is_finger_down(hand, tips[4], pips[4])

def peace_gesture(landmarks, tips, pips):
  hand = landmarks.landmark

  return is_finger_down(hand, tips[0], pips[0]) and \
          not is_finger_down(hand, tips[1], pips[1]) and \
          not is_finger_down(hand, tips[2], pips[2]) and \
          is_finger_down(hand, tips[3], pips[3]) and \
          is_finger_down(hand, tips[4], pips[4])

def surfer_gesture(landmarks, tips, pips, mcps):
  hand = landmarks.landmark

  thumb_up = hand[tips[0]].y < hand[mcps[0]].y 

  pinky_extended = hand[tips[4]].x < hand[pips[4]].x 

  fingers_bent = all(
        hand[tips[i]].x > hand[pips[i]].x
        for i in range(1, 3)
    )
  
  return thumb_up and fingers_bent and pinky_extended and hand_turned(landmarks, mcps)

def punch_gesture(landmarks, tips, pips, mcps):
  hand = landmarks.landmark

  hand_front = all(
     hand[mcps[i]].y < hand[pips[i]].y
     for i in range(1,4)
  )

  return hand[mcps[1]].x < hand[mcps[4]].x and hand_front and hand[pips[2]].y < hand[tips[0]].y

def promise_gesture(landmarks, tips, pips):
  hand = landmarks.landmark

  finger_crossed = hand[tips[1]].x > hand[tips[2]].x

  return finger_crossed and is_finger_down(hand, tips[0], pips[0]) and is_finger_down(hand, tips[3], pips[3]) and is_finger_down(hand, tips[4], pips[4])

def italian_gesture(landmarks, tips):
  hand = landmarks.landmark

  return are_touching(hand, tips[0], tips[1]) and \
          are_touching(hand, tips[1], tips[2]) and \
          are_touching(hand, tips[2], tips[3]) and \
          are_touching(hand, tips[3], tips[4]) and \
          are_touching(hand, tips[4], tips[0])

def tiny_gesture(landmarks, tips, pips):
  hand = landmarks.landmark

  return abs(hand[tips[0]].y - hand[tips[1]].y) > 0.05 and \
          abs(hand[tips[0]].y - hand[tips[1]].y) < 0.1 and \
          is_finger_down(hand, tips[2], pips[2]) and \
          is_finger_down(hand, tips[3], pips[3]) and \
          is_finger_down(hand, tips[4], pips[4])

def pistol_gesture(landmarks, tips, pips, mcps):
  hand = landmarks.landmark

  thumb_up = hand[tips[0]].y < hand[mcps[0]].y
  index_finger_extended = hand[tips[1]].x < hand[mcps[1]].x

  return hand_turned(landmarks, mcps) and \
          is_finger_down(hand, tips[2], pips[2]) and \
          is_finger_down(hand, tips[3], pips[3]) and \
          is_finger_down(hand, tips[4], pips[4]) and \
          thumb_up and index_finger_extended

def uwu_gesture(landmarks, tips, pips, mcps):
  hand = landmarks.landmark

  finger_bent = all(
    hand[tips[i]].x > hand[pips[i]].x
    for i in range(2, 4)
  )

  thumb_up = hand[tips[0]].y < hand[mcps[0]].y
  index_finger_up = hand[tips[1]].y < hand[mcps[1]].y

  fingers_crossed = hand[tips[0]].x > hand[tips[1]].x

  return finger_bent and thumb_up and index_finger_up and fingers_crossed and hand_flipped(hand, mcps)

def looser_gesture(landmarks, tips, pips):
  hand = landmarks.landmark

  return not is_finger_down(hand, tips[0], pips[0]) and \
          not is_finger_down(hand, tips[1], pips[1]) and \
          is_finger_down(hand, tips[2], pips[2]) and \
          is_finger_down(hand, tips[3], pips[3]) and \
          is_finger_down(hand, tips[4], pips[4])

def friki_gesture(landmarks, tips, pips):
  hand = landmarks.landmark

  return  is_finger_down(hand, tips[0], pips[0]) and \
          not is_finger_down(hand, tips[1], pips[1]) and \
          is_finger_down(hand, tips[2], pips[2]) and \
          is_finger_down(hand, tips[3], pips[3]) and \
          is_finger_down(hand, tips[4], pips[4])

def vulcano_gesture(landmarks, tips, pips):
  hand = landmarks.landmark

  return is_hand_open(hand, tips, pips) and \
          abs(hand[tips[2]].x - hand[tips[3]].x) > 0.08 and \
          abs(hand[tips[1]].x - hand[tips[2]].x) < 0.05 and \
          abs(hand[tips[3]].x - hand[tips[4]].x) < 0.05


gesture_images = load_gestures_images()

for gesture, img in gesture_images.items():
    if img is None:
        print(f"Failed to load image for: {gesture}")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5,
                    max_num_hands = 1) as hands:

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    current_gesture = "no_gesture"

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame_display = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
      results_hands = hands.process(frame_display)  

      if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
          mp_drawing.draw_landmarks(frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
          
          finger_tips = [4, 8, 12, 16, 20]
          finger_dips = [3, 7, 11, 15, 19]
          finger_pips = [2, 6, 10, 14, 18]
          finger_mcps = [1, 5, 9, 13, 17]

          if is_hand_open(hand_landmarks.landmark, finger_tips, finger_pips) and \
             not hand_turned(hand_landmarks, finger_mcps) and \
             not rock_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not yey_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not fuck_you(hand_landmarks, finger_tips, finger_pips, finger_mcps) and \
             not perfect_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not peace_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not bad_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps) and \
             not punch_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps) and \
             not promise_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not italian_gesture(hand_landmarks, finger_tips) and \
             not tiny_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not uwu_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps) and \
             not looser_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not friki_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not vulcano_gesture(hand_landmarks, finger_tips, finger_pips):
            current_gesture = "givemefive"
            # print('Give me five')

          elif okey_gesture(hand_landmarks, finger_mcps, finger_tips, finger_pips) and \
            not surfer_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
            current_gesture = "okey"
            # print('Okeyy')

          elif not is_hand_open(hand_landmarks.landmark, finger_tips, finger_pips):
            current_gesture = "blm"
            # print('Black lives matter')
            
          elif rock_gesture(hand_landmarks, finger_tips, finger_pips):
            current_gesture = "rock"
            # print('Rock')
          
          elif yey_gesture(hand_landmarks, finger_tips, finger_pips):
            current_gesture = "yey"
            # print('Yey')
          
          elif fuck_you(hand_landmarks, finger_tips, finger_pips, finger_mcps):
            current_gesture = "fucku"
            # print('Fuck you')

          elif perfect_gesture(hand_landmarks, finger_tips, finger_pips) and \
            not italian_gesture(hand_landmarks, finger_tips) and \
            not tiny_gesture(hand_landmarks, finger_tips, finger_pips):
              current_gesture = "perfect"
              # print("Perfect")

             
          elif peace_gesture(hand_landmarks, finger_tips, finger_pips) and \
             not promise_gesture(hand_landmarks, finger_tips, finger_pips):
             current_gesture = "peace"
            #  print("Peace")


          elif surfer_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
             current_gesture = "surf"
            #  print("Surfffff")


          elif bad_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
             current_gesture = "bad"
            #  print("Badd")


          elif punch_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
             current_gesture = "punch"
            #  print("Punch")


          elif promise_gesture(hand_landmarks, finger_tips, finger_pips):
             current_gesture = "promise"
            #  print("Promise")


          elif italian_gesture(hand_landmarks, finger_tips) and \
            not uwu_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
             current_gesture = "italian"
            #  print("Italian")


          elif tiny_gesture(hand_landmarks, finger_tips, finger_pips):
              current_gesture = "tiny"
              # print("la tienes pequeña")


          elif pistol_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps) and \
            not uwu_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
              current_gesture = "pistol"
              # print("Pistol")

          
          elif uwu_gesture(hand_landmarks, finger_tips, finger_pips, finger_mcps):
              current_gesture = "uwu"
              # print("UwU")


          elif looser_gesture(hand_landmarks, finger_tips, finger_pips):
              current_gesture = "looser"
              # print("Looser")

          
          elif friki_gesture(hand_landmarks, finger_tips, finger_pips):
              current_gesture = "friki"
              # print("Friki")


          elif vulcano_gesture(hand_landmarks, finger_tips, finger_pips):
              current_gesture = "vulcano"
              # print("Vulcano")

          else:
              current_gesture = "no_gesture"
              # print('No gesture detected')

      if current_gesture in gesture_images and gesture_images[current_gesture] is not None:
        frame_display = overlay_image(frame_display, gesture_images[current_gesture], 
                                      pos=(25, 70))
      
      cv2.putText(frame_display, "TIENE QUE SER LA MANO DERECHA", (80,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,204,0), 5)
      cv2.imshow('Hand Tracking', cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))

      if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

