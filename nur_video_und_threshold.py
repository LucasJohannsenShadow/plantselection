#import cv2.cv2 as cv3
import cv2 as cv3
import cv2
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective

import numpy as np

import imutils


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

# realsence d435
cap = cv3.VideoCapture(0)
while True:
    ret, frame = cap.read()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # store the a-channel
    a_channel = lab[:, :, 1]
    # Automate threshold using Otsu method
    th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Mask the result with the original image
    # masked = cv2.bitwise_and(frame, frame, mask=th)
    contours, _ = cv2.findContours(image=th, mode=cv3.RETR_TREE, method=cv3.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv3.contourArea(contour), contour) for contour in contours]

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(th, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    contours = cv2.findContours(edged.copy(), cv3.RETR_EXTERNAL, cv3.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # 'pixels per metric' calibration variable
    # wie viele Pixel machen ein cm aus
    pixelsPerMetric = 16.78260869565217391304347826087
    # loop over the contours individually
    # loop through the contours

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            area = cv2.contourArea(cnt)
            cv2.drawContours(frame, [cnt], 0, 255, -1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            Moments = cv2.moments(cnt)
            x = int(Moments["m10"] / Moments["m00"])
            y = int(Moments["m01"] / Moments["m00"])
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(frame, "{:.1f} pixelQ".format(area), (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

    for c in contours:
        if cv2.contourArea(c) > 1000:
            # compute the rotated bounding box of the contour
            frame = frame.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            # loop over the original points and draw them
            #    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # draw the midpoints on the image
            cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 255, 0), -1)
            cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 255, 0), -1)
            cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 255, 0), -1)
            cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 255, 0), -1)
            # draw lines between the midpoints
            cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 1)
            cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 1)
            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            vergleichsgrosse = 0.05927835051546391752577319587629
            #if pixelsPerMetric is None:
             #   pixelsPerMetric = dB / vergleichsgrosse
            # compute the size of the object
            dimA = dA/pixelsPerMetric
            dimB = dB/pixelsPerMetric
            # draw the object sizes on the image
            cv2.putText(frame, "{:.1f} cm".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(frame, "{:.1f} cm".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
    cv3.imshow('orf', frame)
    print('Total number of contours detected: ' + str(len(contours)))
    # cv3.imshow('th', th)
    if cv3.waitKey(1) == ord('q'):
        break

cap.release()
cv3.destroyAllWindows()
cap = cv3.VideoCapture(0)
