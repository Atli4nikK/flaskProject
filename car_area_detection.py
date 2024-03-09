import cv2
import numpy as np
import torch
from ultralytics import YOLO
from base_camera import BaseCamera

'''
def draw_polygon(event, x, y, flags, param):
    global polygon, drawing, ix, iy, redraw
    x = int(x * frame.shape[1] / frame_orig.shape[1])
    y = int(y * frame.shape[0] / frame_orig.shape[0])

    if redraw:
        polygon = []
        drawing = False
        redraw = False

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        polygon.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(frame, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        polygon.append((x, y))

    return frame


def draw_white_bbox():
    # cv2.circle(frame_orig, (x1_orig, y1_orig), 2, (255, 255, 255), -1)
    # cv2.circle(frame_orig, (x2_orig, y1_orig), 2, (255, 255, 255), -1)
    # cv2.circle(frame_orig, (x2_orig, y2_orig), 2, (255, 255, 255), -1)
    # cv2.circle(frame_orig, (x1_orig, y2_orig), 2, (255, 255, 255), -1)

    overlay = frame_orig.copy()
    cv2.rectangle(overlay, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 255, 255), -1)
    alpha = 0.3
    cv2.addWeighted(frame_orig, 1 - alpha, overlay, alpha, 0, frame_orig)

    class_label = "car"
    cv2.putText(frame_orig, class_label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

    # up_left
    cv2.line(frame_orig, (x1_orig, y1_orig), (x1_orig + 10, y1_orig), (255, 255, 255), 2)
    cv2.line(frame_orig, (x1_orig, y1_orig), (x1_orig, y1_orig + 10), (255, 255, 255), 2)

    # up_right
    cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y1_orig),
             (x1_orig + (x2_orig - x1_orig) - 10, y1_orig), (255, 255, 255), 2)
    cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y1_orig),
             (x1_orig + (x2_orig - x1_orig), y1_orig + 10), (255, 255, 255), 2)

    # down_right
    cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y2_orig),
             (x1_orig + (x2_orig - x1_orig), y2_orig - 10), (255, 255, 255), 2)
    cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y2_orig),
             (x1_orig + (x2_orig - x1_orig) - 10, y2_orig), (255, 255, 255), 2)

    # down_left
    cv2.line(frame_orig, (x1_orig, y1_orig + (y2_orig - y1_orig)),
             (x1_orig, y1_orig + (y2_orig - y1_orig) - 10), (255, 255, 255), 2)
    cv2.line(frame_orig, (x1_orig, y1_orig + (y2_orig - y1_orig)),
             (x1_orig + 10, y1_orig + (y2_orig - y1_orig)), (255, 255, 255), 2)

'''


class Camera(BaseCamera):
    @staticmethod
    def frames():
        cls = 2
        input_size = 640
        ESC = 27
        stream_source = "test.mp4"
        model_name = 'yolov8s.pt'
        window_name = "Test"
        # TODO:
        # update hud:
        # - class of bbox style DONE optimize, write class above bbox DONE but need to refactor
        # - FPS DONE, instructions, time, etc.
        # object heat map
        # detect object center inside polygon
        # docker
        model = YOLO(model_name)

        cap = cv2.VideoCapture(stream_source)

        polygon = []
        drawing = False
        redraw = False

        # cv2.namedWindow(window_name)
        # cv2.setMouseCallback(window_name, draw_polygon)

        fps = 0
        prev_time = 0

        while True:

            ret, frame_orig = cap.read()

            frame = cv2.resize(frame_orig, (input_size, input_size))

            frame_transposed = np.transpose(frame, (2, 0, 1))

            frame_transposed = np.expand_dims(frame_transposed, 0)

            img = torch.from_numpy(frame_transposed).float().div(255)

            if torch.cuda.is_available():
                img = img.cuda()

            results = model(img)

            car_boxes = results[0].boxes.xyxy[results[0].boxes.cls == cls]

            inside_polygon = False
            for box in car_boxes:

                box = box.cpu().numpy().astype(np.int32)
                x1, y1, x2, y2 = box

                x1_orig = int(x1 * frame_orig.shape[1] / frame.shape[1])
                y1_orig = int(y1 * frame_orig.shape[0] / frame.shape[0])
                x2_orig = int(x2 * frame_orig.shape[1] / frame.shape[1])
                y2_orig = int(y2 * frame_orig.shape[0] / frame.shape[0])

                # cv2.rectangle(frame_orig, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 255, 255), 2)
                # draw_white_bbox()
                overlay = frame_orig.copy()
                cv2.rectangle(overlay, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 255, 255), -1)
                alpha = 0.3
                cv2.addWeighted(frame_orig, 1 - alpha, overlay, alpha, 0, frame_orig)

                class_label = "car"
                cv2.putText(frame_orig, class_label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

                # up_left
                cv2.line(frame_orig, (x1_orig, y1_orig), (x1_orig + 10, y1_orig), (255, 255, 255), 2)
                cv2.line(frame_orig, (x1_orig, y1_orig), (x1_orig, y1_orig + 10), (255, 255, 255), 2)

                # up_right
                cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y1_orig),
                         (x1_orig + (x2_orig - x1_orig) - 10, y1_orig), (255, 255, 255), 2)
                cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y1_orig),
                         (x1_orig + (x2_orig - x1_orig), y1_orig + 10), (255, 255, 255), 2)

                # down_right
                cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y2_orig),
                         (x1_orig + (x2_orig - x1_orig), y2_orig - 10), (255, 255, 255), 2)
                cv2.line(frame_orig, (x1_orig + (x2_orig - x1_orig), y2_orig),
                         (x1_orig + (x2_orig - x1_orig) - 10, y2_orig), (255, 255, 255), 2)

                # down_left
                cv2.line(frame_orig, (x1_orig, y1_orig + (y2_orig - y1_orig)),
                         (x1_orig, y1_orig + (y2_orig - y1_orig) - 10), (255, 255, 255), 2)
                cv2.line(frame_orig, (x1_orig, y1_orig + (y2_orig - y1_orig)),
                         (x1_orig + 10, y1_orig + (y2_orig - y1_orig)), (255, 255, 255), 2)

                pt1 = (x1, y1)
                pt2 = (x2, y1)
                pt3 = (x2, y2)
                pt4 = (x1, y2)

                # refactor to center bbox checking
                if len(polygon) > 3:
                    if cv2.pointPolygonTest(np.array(polygon), (int(pt1[0]), int(pt1[1])), False) >= 0 and \
                            cv2.pointPolygonTest(np.array(polygon), (int(pt2[0]), int(pt2[1])), False) >= 0 and \
                            cv2.pointPolygonTest(np.array(polygon), (int(pt3[0]), int(pt3[1])), False) >= 0 and \
                            cv2.pointPolygonTest(np.array(polygon), (int(pt4[0]), int(pt4[1])), False) >= 0:
                        inside_polygon = True
                        break

            if len(polygon) > 3:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                pts_orig = pts * [frame_orig.shape[1] / frame.shape[1], frame_orig.shape[0] / frame.shape[0]]
                pts_orig = np.array(pts_orig, np.int32)
                if inside_polygon:
                    cv2.polylines(frame_orig, [pts_orig], True, (0, 255, 0), 2)
                else:
                    cv2.polylines(frame_orig, [pts_orig], True, (0, 0, 255), 2)

            curr_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(frame_orig, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv2.imshow(window_name, frame_orig)
            yield cv2.imencode('.jpg', frame_orig)[1].tobytes()
