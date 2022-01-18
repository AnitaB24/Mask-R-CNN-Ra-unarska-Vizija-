import numpy as np
import argparse
import time
import cv2
import imutils
import os

# parsirati argumente komandne linije
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="ulazna putanja slike")
ap.add_argument("-m", "--mask-rcnn", required=True, help="putanja do mask-rcnn direktorijuma")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimalna verovatnoca filtriranja")
args = vars(ap.parse_args())

# ucitavanje liste labela klasa COCO skupa podataka
labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# inicijalizacija liste boja za predstavljanje svake labele klase
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# inicijalizacija putanja do parametara istreniranog Mask R-CNN modela
weightsPath = os.path.sep.join([args["mask_rcnn"], "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# ucitavanje Mask R-CNN sa diska
print("INFO: Ucitavanje Mask R-CNN sa diska")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# ucitavanje slike i dimenzija slike
orig = cv2.imread(args["image"])
(H, W) = orig.shape[:2]

# inicijalizacija kopije na kojoj ce se iscrtavati rezultati
image = orig.copy()

# pretvaranje slike u binarni objekat tipa BGRA
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
# postavljanje ulaza u Mask R-CNN
net.setInput(blob)

# racunanje okvirnih pravougaionika, rezultujucih maski i odredjivanje
# vremena potrebnog da se racunanje izvrsi
start = time.time()
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
end = time.time()

# prikazivanje vremena potrebnog za izracunavanje
print("INFO: Vreme potrebno za izracunavanje {:.2f}s".format(end - start))

# provera da li ima prepoznatih objekata
if boxes.shape[2]:
    # prolazak kroz niz prepoznatih objekata
    for i in range(0, boxes.shape[2]):

        # preuzimanje ID-ja i verovatnoce detektovane klase
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filtriranje na osnovu ulaznog argumenta verovatnoce
        if confidence > args["confidence"]:
            # skaliranje okvirnog pravougaonika na osnovu originalne velicine slike
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # izdvajanje i skaliranje maske objekta koja se
            # interpolacijom konvertuje u binarnu masku
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.2)

            # izdvajanje maskiranog regiona sa objektom
            roi = image[startY:endY, startX:endX][mask]

            # izabrati boju koja ce se koristi za vizuelizaciju i izmesati je
            # sa maskiranim regionom
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # nalepiti izdvojeni region nazad u kopiju originalne slike
            image[startY:endY, startX:endX][mask] = blended

            # iscrtati okvirni pravougaonik regiona i ispisati labelu klase i verovatnocu
            color = [int(c) for c in color]
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.3f}".format(LABELS[classID], confidence)
            cv2.putText(image, text, (startX, startY - 5), cv2.QT_FONT_NORMAL, 0.5, color, 2)

    # prikazati originalnu sliku i sliku sa prepoznatim objektima
    cv2.imshow("Originalna Slika", imutils.resize(orig, height = 650))
    cv2.imshow("Rezultujuca Slika", imutils.resize(image, height = 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ako nema prepoznatih objekata
else:
    print("INFO: Nije prepoznat nijedan objekat.")