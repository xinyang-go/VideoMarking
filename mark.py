import cv2
import os
import numpy as np
import argparse

id2name = {
    (0, 0): "R1",
    (0, 1): "R2",
    (0, 2): "R3",
    (0, 3): "R4",
    (0, 4): "R5",
    (0, 5): "R7",
    (0, 6): "R8",
    (1, 0): "B1",
    (1, 1): "B2",
    (1, 2): "B3",
    (1, 3): "B4",
    (1, 4): "B5",
    (1, 5): "B7",
    (1, 6): "B8",
    (2, 0): "N1",
    (2, 1): "N2",
    (2, 2): "N3",
    (2, 3): "N4",
    (2, 4): "N5",
    (2, 5): "N7",
    (2, 6): "N8",
}
font = cv2.FONT_HERSHEY_DUPLEX


def showImgBoxes(img, box, boxes, name="frame"):
    img2show = img.copy()
    if boxes:
        for i, b in enumerate(boxes):
            img2show = cv2.rectangle(img2show, tuple(b[3:5]), tuple(b[5:]), (0, 0, 255), 1)
            img2show = cv2.putText(img2show, id2name[tuple(b[1:3])], tuple(b[3:5]), font, 1, (0, 0, 255), 1)
            img2show = cv2.putText(img2show, str(i), tuple(box[3:5]), font, 1, (0, 0, 255), 1)
    if box is not None:
        img2show = cv2.rectangle(img2show, tuple(box[3:5]), tuple(box[5:]), (255, 0, 0), 1)
        img2show = cv2.putText(img2show, id2name[tuple(box[1:3])], tuple(box[3:5]), font, 1, (255, 0, 0), 1)
    cv2.imshow(name, img2show)


def loadExist(fn):
    return np.loadtxt(fn, ndmin=2).astype(np.int).tolist() if os.path.isfile(fn) else []


def updateBox(img, tracker, box):
    if box is not None and tracker is not None:
        ok, b = tracker.update(img)
        if not ok:
            box = None
            tracker = None
        else:
            box = [box[0], box[1], box[2], int(b[0]), int(b[1]), int(b[2]), int(b[3])]
    return tracker, box


def mark(vn, dir):
    keyVal = {
        "UP": 1113938,
        "DOWN": 1113940,
        "LEFT": 1113937,
        "RIGHT": 1113939,
        "w": 1048695,
        "a": 1048673,
        "s": 1048691,
        "d": 1048676,
        " ": 1048608,
        "q": 1048689,
        "r": 1048690,
        "x": 1048696,
        'j': 1048682,
        '0': 1048624,
        '9': 1048633,
        '\n': 1048589
    }

    jump = 0
    vname = os.path.basename(vn)
    video = cv2.VideoCapture(vn)
    row = int(video.get(4))
    col = int(video.get(3))
    print(row, col)
    boxes = []
    tracker = None
    box = None
    cls = 0
    _, img = video.read()
    cnt = 0
    while _:
        boxes = loadExist(f"{dir}/{vname}-{cnt}.txt")
        tracker, box = updateBox(img, tracker, box)
        showImgBoxes(img, box, boxes)
        k = cv2.waitKeyEx(jump)
        # print(k)
        if jump:
            if k == keyVal['j']:
                jump = 0
        else:
            while k != keyVal[" "]:
                if k == keyVal['j']:
                    jump = 7
                    break
                if keyVal['0'] <= k <= keyVal['9']:
                    cls = k - keyVal['0']
                    k = cv2.waitKeyEx(0)
                    while k != keyVal['\n']:
                        if keyVal['0'] <= k <= keyVal['9']:
                            cls = cls * 10 + k - keyVal['0']
                        k = cv2.waitKeyEx(0)
                    print("\nchange label to ", cls)
                elif k == keyVal['x']:
                    p = 0
                    while k != keyVal['\n']:
                        if keyVal['0'] <= k <= keyVal['9']:
                            p = p * 10 + k - keyVal['0']
                        k = cv2.waitKeyEx(0)
                    if p >= len(boxes):
                        print("idx out of range!")
                        continue
                    tmp = box
                    box = boxes[p]
                    if tmp is None:
                        del boxes[p]
                    else:
                        boxes[p] = tmp
                elif box is None:
                    if k == keyVal["r"]:
                        box = cv2.selectROI(windowName="frame", img=img, showCrosshair=True, fromCenter=False)
                        box = [0, cls // 10, cls % 10, int(box[0]), int(box[1]), int(box[0] + box[2]),
                               int(box[1] + box[3])]
                        tracker = cv2.TrackerKCF_create()
                        ok = tracker.init(img, tuple(box[3:]))
                else:
                    if k == keyVal["DOWN"]:
                        box[4] = min(row, box[4] + 2)
                        box[6] = min(row, box[6] + 2)
                    elif k == keyVal["UP"]:
                        box[4] = max(0, box[4] - 2)
                        box[6] = max(0, box[6] - 2)
                    elif k == keyVal["RIGHT"]:
                        box[3] = min(col, box[3] + 2)
                        box[5] = min(col, box[5] + 2)
                    elif k == keyVal["LEFT"]:
                        box[3] = max(0, box[3] - 2)
                        box[5] = max(0, box[5] - 2)
                    elif k == keyVal["w"]:
                        box[4] = max(0, box[4] - 2)
                        box[6] = min(row, box[6] + 2)
                    elif k == keyVal["s"]:
                        box[4] = min(box[6] - 2, box[4] + 2)
                        box[6] = max(box[4], box[6] - 2)
                    elif k == keyVal["d"]:
                        box[3] = max(0, box[3] - 2)
                        box[5] = min(col, box[5] + 2)
                    elif k == keyVal["a"]:
                        box[3] = min(box[5] - 2, box[3] + 2)
                        box[5] = max(box[3], box[5] - 2)
                    elif k == keyVal["r"]:
                        box = cv2.selectROI(windowName="frame", img=img, showCrosshair=True, fromCenter=False)
                        box = [0, cls // 10, cls % 10, int(box[0]), int(box[1]), int(box[0] + box[2]),
                               int(box[1] + box[3])]
                    elif k == keyVal["q"]:
                        box = None
                        tracker = None
                    if box and tracker:
                        tracker = cv2.TrackerKCF_create()
                        ok = tracker.init(img, tuple(box[3:]))
                showImgBoxes(img, box, boxes)
                k = cv2.waitKeyEx(0)
                # print(k)
            if k == keyVal[" "]:
                if box is not None and box[3:7] != [0, 0, 0, 0]:
                    boxes.append(box)
                if boxes:
                    np.savetxt(f"{dir}/{vname}-{cnt}.txt", boxes)
        _, img = video.read()
        cnt += 1


def saveImgBoxes(vn, dir):
    vname = os.path.basename(vn)
    video = cv2.VideoCapture(vn)
    _, img = video.read()
    cnt = 0
    while _:
        boxes = loadExist(f"{dir}/{vname}-{cnt}.txt")
        if boxes:
            cv2.imwrite(f"{dir}/{vname}-{cnt}.jpg", img)
        _, img = video.read()
        cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="The video you want to mark.")
    parser.add_argument("--dir", type=str, help="The dir you want to save the labels.")
    parser.add_argument("--save", type=bool, default=False, help="set to true if already finishing marking")
    opt = parser.parse_args()

    if opt.save:
        saveImgBoxes(opt.video, opt.dir)
    else:
        mark(opt.video, opt.dir)
