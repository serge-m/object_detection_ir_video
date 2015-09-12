#!/usr/bin/python
"""
Manual labeling tool. Use to create training set for object detection algorithm.
"""
# import the necessary packages
import argparse
import cv2
import itertools
import os
import csv
import logging
import logging.config
import json 


def setup_logging():
    """
    Setup logging module using 'logging_config.json' configuration file
    :return:
    """
    name_json = 'logging_config.json'
    path_json = os.path.join(os.path.dirname(__file__), name_json)
    with open(path_json, 'r') as f_json:
        dict_config = json.load(f_json)
    logging.config.dictConfig(dict_config)


class ImgNames(object):
    set_supported_ext = ['.jpg', '.png', '.bmp']
    def __init__(self, path_img):
        self.path_start = path_img
        self.path_dir, fname_start = os.path.split(path_img)
        walk = os.walk(self.path_dir)
        _, _, self.list_files = walk.next()

        self.list_files = filter(lambda name: os.path.splitext(name)[1] in self.set_supported_ext, self.list_files)
        self.list_files.sort()
        self.idx_cur = self.list_files.index(fname_start)

    def get(self, step=0):
        logger = logging.getLogger(__name__)
        self.idx_cur += step
        path = os.path.join(self.path_dir, self.list_files[self.idx_cur])
        logger.info("Selected id {}, file {}".format(self.idx_cur,path ))
        return path


class ImageWithROI(object):
    def __init__(self):
        self.path_img = None
        self.image = None
        self.image_clean = None
        self.list_rect = []
        self.path_labels = None

    @staticmethod
    def load_csv(path_csv):
        if not os.path.exists(path_csv):
            return []

        with open(path_csv, "rb") as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='"')
            return [map(int, row, ) for row in reader]

    @staticmethod
    def save_csv(path_csv, list_rect):
        with open(path_csv, "wb") as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='"')
            writer.writerows(list_rect)
            
                    

    def load(self, path_img, prefix_labels=".labels2.txt"):
        logger = logging.getLogger(__name__)
        logger.info("Trying to load image {}".format(path_img))
        image = cv2.imread(path_img)
        if image is None:
            logger.warning("Loading failed. Staying on the current image {}".format(self.path_img))
            return 
        
        self.path_img = path_img
        self.image = image
        self.image_clean = self.image.copy()
        self.path_labels = self.path_img + prefix_labels
        self.list_rect = self.load_csv(self.path_labels)
        for rect in self.list_rect:
            cv2.rectangle(self.image, tuple(rect[0:2]), tuple(rect[2:4]), (0, 255, 0), 2)
        logger.debug("loaded  {} rects" .format(len(self.list_rect)))

    def add_rect(self, rect):
        logger = logging.getLogger(__name__)
        logger.debug("add_rect, rect {}".format(rect))
        if rect is None:
            return
        self.list_rect.append(rect)
        cv2.rectangle(self.image, tuple(rect[0:2]), tuple(rect[2:4]), (0, 255, 0), 2)

    def get_image(self, rect = None):
        logger = logging.getLogger(__name__)
        logger.debug("{} -> get_image. {}".format(self, rect))
        if rect is None:
            return self.image
        return self.image[rect[1]:rect[3], rect[0]:rect[2],]

    def clear_image(self):
        self.list_rect = []
        self.image = self.image_clean.copy()


    def save(self):
        logger = logging.getLogger(__name__)
        
        if self.path_labels is not None:
            self.save_csv(self.path_labels, self.list_rect)
            logger.debug("saved list to {}".format(self.path_labels))
        else:
            if self.list_rect:
                logger.warning("self.path_labels is empty, list_rect {}".format(self.list_rect))


# # initialize the list of reference points and boolean indicating
# # whether cropping is being performed or not
# refPt = []
# cropping = False
 
# def click_and_crop(event, x, y, flags, param):
#     # grab references to the global variables
#     global refPt, cropping
 
#     # if the left mouse button was clicked, record the starting
#     # (x, y) coordinates and indicate that cropping is being
#     # performed
#     if event == cv2.EVENT_LBUTTONDOWN:
#         refPt = [(x, y)]
#         cropping = True
 
#     # check to see if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates and indicate that
#         # the cropping operation is finished
#         refPt.append((x, y))
#         cropping = False
 
#         # draw a rectangle around the region of interest
#         cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
#         cv2.imshow("image", image)



class DisplayImage(object):
    def __init__(self, name_main_window, image_container):
        self.selection = []
        self.cropping = False
        self.name_main_window = name_main_window
        cv2.namedWindow(self.name_main_window)

        self.image_container = image_container
        self.image = self.image_container.get_image()
        
        

        def click_and_crop(event, x, y, flags, param):
            logger = logging.getLogger(__name__)
            if event == cv2.EVENT_LBUTTONDOWN:
                logger.debug("Button down {}".format((x,y)))
                self.selection = [(x, y)]
                self.cropping = True
                self.image = self.image_container.get_image().copy()
                cv2.imshow(self.name_main_window, self.image)
         
            # check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                logger.debug("Button up {}".format((x,y)))
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                self.selection.append((x, y))
                self.cropping = False
         
                # draw a rectangle around the region of interest
                cv2.rectangle(self.image, self.selection[0], self.selection[1], (0, 255, 0), 2)
                cv2.imshow(self.name_main_window, self.image)


        cv2.setMouseCallback(self.name_main_window, click_and_crop)

    def update_image(self):
        self.image = self.image_container.get_image().copy()
        self.show_image()

    def show_image(self):
        self.cropping = False
        self.selection = []
        cv2.imshow(self.name_main_window, self.image)
        cv2.destroyWindow("ROI")

    def get_rect(self):
        if len(self.selection) == 2:
            return list(itertools.chain.from_iterable(self.selection))
        return None
        
        

if __name__ == '__main__':
    setup_logging()
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
     
    # load the image, clone it, and setup the mouse callback function
    
    imgnames = ImgNames(args["image"])

    img_container = ImageWithROI()
    img_container.load(imgnames.get())
    
    display_image = DisplayImage("image", img_container)
     
    display_image.show_image()
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        
        key = cv2.waitKey(1) & 0xFF
     
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            img_container.clear_image()
            display_image.update_image()
            
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            rect = display_image.get_rect()
            img_container.add_rect(rect)
            roi = img_container.get_image(rect)
            if roi is not None:
                cv2.imshow("ROI", roi)

        elif key == ord("q"):
            break
        elif key == ord("s"):
            img_container.save()    
        elif key ^ 0xFF and key == 85:
            img_container.load(imgnames.get(+1))
            display_image.update_image()
        elif key ^ 0xFF and key == 86:
            img_container.load(imgnames.get(-1))
            display_image.update_image()


    
             
    cv2.destroyAllWindows()