import numpy as np
import cv2
import duckietown_utils as dtu

from collections import namedtuple
from skimage.morphology import skeletonize_3d


Detections = namedtuple('Detections', 'white yellow red')


class LineDetectorHSV(dtu.Configurable):
    def __init__(self, configuration):
        params_names = [
            'hsv_white1', 'hsv_white2',
            'hsv_yellow1', 'hsv_yellow2',
            'hsv_red1', 'hsv_red2',
            'hsv_red3', 'hsv_red4',
            'hsv_black1', 'hsv_black2'

            'kernel_size',
            'large_kernel_size'
        ]
        super(LineDetectorHSV, self).__init__(params_names, configuration)

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size))
        self._large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (self.large_kernel_size, self.large_kernel_size))

    def color_filter(self, hsv_image, road_mask):
        # The masks are eroded with a small kernel to remove their edges
        # because they are all combined into a single binary image to only
        # call skeletonize once (for efficiency).

        # Filter white
        white_mask = cv2.inRange(hsv_image, self.hsv_white1, self.hsv_white2)
        # white_mask = cv2.dilate(white_mask, self._kernel, iterations=1)
        # white_mask = cv2.erode(white_mask, self._kernel, iterations=2)
        # white_mask = cv2.bitwise_and(white_mask, road_mask)

        # Filter yellow
        yellow_mask = cv2.inRange(hsv_image, self.hsv_yellow1, self.hsv_yellow2)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, self._large_kernel)
        # yellow_mask = cv2.erode(yellow_mask, self._kernel, iterations=1)

        # Filter red
        red_mask_1 = cv2.inRange(hsv_image, self.hsv_red1, self.hsv_red2)
        red_mask_2 = cv2.inRange(hsv_image, self.hsv_red3, self.hsv_red4)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        # red_mask = cv2.dilate(red_mask, self._kernel, iterations=1)
        # red_mask = cv2.erode(red_mask, self._kernel, iterations=2)

        return Detections(white=white_mask, yellow=yellow_mask, red=red_mask)

    def get_road_mask(self, hsv_image):
        black_mask = cv2.inRange(hsv_image, self.hsv_black1, self.hsv_black2)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, self._large_kernel)

        # Fill the bottom of the mask to merge unconnected components
        # and the top to remove artifacts from opening
        connected_mask = black_mask.copy()
        connected_mask[:self.large_kernel_size] = 0
        connected_mask[-self.large_kernel_size:] = 255

        # Use flood fill to detect the largest component (corresponding to the road)
        height, width = black_mask.shape
        road_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(connected_mask, road_mask, (0, height - 1), 255)

        # Filter out the non-road components and dilate it
        road_mask = cv2.bitwise_and(black_mask, road_mask[1:-1, 1:-1])
        road_mask = cv2.dilate(road_mask, self._kernel, iterations=2)

        return road_mask

    def get_connected_skeletons(self, skeletons, num_components, components):
        x_white, y_white = np.nonzero(skeletons.white)
        indices_white = [[] for _ in range(num_components)]
        for index, (x, y) in enumerate(zip(x_white, y_white)):
            indices_white[components[x, y]].append(index)

        x_yellow, y_yellow = np.nonzero(skeletons.yellow)
        indices_yellow = [[] for _ in range(num_components)]
        for index, (x, y) in enumerate(zip(x_yellow, y_yellow)):
            indices_yellow[components[x, y]].append(index)

        x_red, y_red = np.nonzero(skeletons.red)
        indices_red = [[] for _ in range(num_components)]
        for index, (x, y) in enumerate(zip(x_red, y_red)):
            indices_red[components[x, y]].append(index)

        return Detections(white=[(x_white[indices], y_white[indices])
                            for indices in indices_white if indices],
                          yellow=[(x_yellow[indices], y_yellow[indices])
                            for indices in indices_yellow if indices],
                          red=[(x_red[indices], y_red[indices])
                            for indices in indices_red if indices])

    def get_skeletons(self, masks):
        # Combine all the masks into a single binary image
        binary_image = cv2.bitwise_or(masks.white, masks.yellow)
        binary_image = cv2.bitwise_or(binary_image, masks.red)

        # Get the skeleton, based on [Lee94]
        skeleton = skeletonize_3d(binary_image)

        # Get dilated version of the skeleton to find the connected components.
        # This is to perform a poor man's version of DBSCAN.
        neighbors = cv2.dilate(skeleton, self._large_kernel, iterations=1)
        # Get the connected components
        num_components, components = cv2.connectedComponents(neighbors,
                                                             connectivity=4)

        skeletons = Detections(white=cv2.bitwise_and(skeleton, masks.white),
                               yellow=cv2.bitwise_and(skeleton, masks.yellow),
                               red=cv2.bitwise_and(skeleton, masks.red))

        return self.get_connected_skeletons(skeletons,
                                            num_components,
                                            components)

    def detect_lines(self, bgr_image):
        # Convert the BGR image to HSV
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        # Get the road mask
        road_mask = self.get_road_mask(hsv_image)
        # Filter the colors
        color_masks = self.color_filter(hsv_image)
        # Get the skeletons
        skeletons = self.get_skeletons(color_masks)

        return skeletons, (color_masks, road_mask)
