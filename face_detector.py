import cv2

def getFacesBoundaries(imagePath, faceCascade):
    sourceImage = cv2.imread(imagePath)
    grayImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        grayImage,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    );
    
    croppedImages = []
    
    for (x, y, w, h) in faces:
        croppedImages.append(sourceImage[y:y+h,x:x+w])
        
    return croppedImages;
