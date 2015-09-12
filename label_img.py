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


_button_reset = "r"
_button_add = "a"
_button_quit = "q"
_button_save = "s"
_postfix_labels_file = ".labels.txt"

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
    """
    Storage for images names in a directory.
    The list of files is created once at the initialization.
    """
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
        """
        Navigation between files in the directory.
        :param step: integer step (can be positive, zero and negative)
        :return: path to the corresponding image file
        """
        logger = logging.getLogger(__name__)
        self.idx_cur += step
        path = os.path.join(self.path_dir, self.list_files[self.idx_cur])
        logger.info("Selected id {}, file {}".format(self.idx_cur,path ))
        return path


class ImageWithROI(object):
    """
    Stores image with selected regions on it.
    """
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
            
    def load(self, path_img, postfix_labels=_postfix_labels_file):
        """
        Loads new image to the container. If there is also labels file along with the image, then
        load also stored regions.
        Labels files are assumed to have names like <path_img><postfix_labels>.

        :param path_img: image to load
        :param postfix_labels: postfix for labels files
        :return: None
        """
        logger = logging.getLogger(__name__)
        logger.info("Trying to load image {}".format(path_img))
        image = cv2.imread(path_img)
        if image is None:
            logger.warning("Loading failed. Staying on the current image {}".format(self.path_img))
            return 
        
        self.path_img = path_img
        self.image = image
        self.image_clean = self.image.copy()
        self.path_labels = self.path_img + postfix_labels
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

    def get_image(self, rect=None):
        """
        Returns stored image of patch of the image defined by ROI
        :param rect: ROI to load. None by default
        :return: image of part of the image
        """
        logger = logging.getLogger(__name__)
        logger.debug("{} -> get_image. {}".format(self, rect))
        if rect is None:
            return self.image
        return self.image[rect[1]:rect[3], rect[0]:rect[2],]

    def clear_image(self):
        self.list_rect = []
        self.image = self.image_clean.copy()

    def save(self):
        """
        Save stored regions to labels file
        :return:
        """
        logger = logging.getLogger(__name__)
        
        if self.path_labels is not None:
            self.save_csv(self.path_labels, self.list_rect)
            logger.debug("saved list to {}".format(self.path_labels))
        else:
            if self.list_rect:
                logger.warning("self.path_labels is empty, list_rect {}".format(self.list_rect))


class GUIMarkup(object):
    """
    Class of GUI for image markup.
    """
    def __init__(self, name_main_window, image_container):
        """
        Initialize window, set callbacks
        :param name_main_window: string, name of the window
        :param image_container: must implement get_image() method
        :return:
        """
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
        """
        Returns selection as rectangle
        :return: (x1,y1,x2,y2) tuple if rectangle is selected, None otherwise
        """
        if len(self.selection) == 2:
            return list(itertools.chain.from_iterable(self.selection))
        return None
        

def interactive_labeling(args):
    """
    Main loop of GUI
    :param args: dictionary with parameters
    :return:
    """
    img_names = ImgNames(args["image"])

    img_container = ImageWithROI()
    img_container.load(img_names.get())

    gui = GUIMarkup("image", img_container)

    gui.show_image()
    # keep looping until the 'q' key is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord(_button_reset):
            img_container.clear_image()
            gui.update_image()
        elif key == ord(_button_add):
            rect = gui.get_rect()
            img_container.add_rect(rect)
            roi = img_container.get_image(rect)
            if roi is not None:
                cv2.imshow("ROI", roi)
        elif key == ord(_button_quit):
            break
        elif key == ord(_button_save):
            img_container.save()
        elif key ^ 0xFF and key == 85:
            img_container.load(img_names.get(+1))
            gui.update_image()
        elif key ^ 0xFF and key == 86:
            img_container.load(img_names.get(-1))
            gui.update_image()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    setup_logging()
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    print """Select regions using mouse\n
General pipeline:
    1) Navigate to frame
    2) Select region
    3) Press "{_button_add}" to add region to list. Repeat steps 2-3 if required.
    4) Press "{_button_save}" to save
Controls:
    {_button_reset} - reset selection
    {_button_add} - add crop (selection) to the list, show cropped region
    {_button_quit} - quit
    {_button_save} - save selection to file
    PageUp, PageDown - navigation""".format(**locals())
     
    interactive_labeling(args)