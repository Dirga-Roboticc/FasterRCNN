import numpy as np
import cv2
import torch
import glob as glob
import os
from model import create_model
from config import NUM_CLASSES
# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(
    './outputs/model8.pth', map_location=device, weights_only=True
))
model.eval()
# directory where all the images are present
DIR_TEST = './voc-fontawsome-dataset/images'
test_images = glob.glob(os.path.join(DIR_TEST, '*.*'))
test_images = [img for img in test_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Test instances: {len(test_images)}")
print(f"Test images: {test_images}")  # Print the list of image paths

# Import CLASSES from config
from config import CLASSES
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.1

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = os.path.basename(test_images[i]).split('.')[0]
    image = cv2.imread(test_images[i])
    
    if image is None:
        print(f"Failed to read image: {test_images[i]}")
        continue
    
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
        cv2.imwrite(f"test_predictions/{image_name}.jpg", orig_image)
        
        print(f"Image {i+1} done... Saved as test_predictions/{image_name}.jpg")
        print('-'*50)

print('TEST PREDICTIONS COMPLETE')
print(f"All predictions saved in the 'test_predictions/' directory")
