from make_datapath import make_datapath_list
from extract_inform_annotation import Anno_xml
from libs import *
from utils.augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,\
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans


class DataTransform():

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([ConvertFromInts(), #convert imf from int to float32
                              ToAbsoluteCoords(), #back annotation to normal type
                              PhotometricDistort(), #change color by random
                              Expand(color_mean),
                              RandomSampleCrop(), #random crop image
                              RandomMirror, #Flip image
                              ToPercentCoords(), #standard annotation in range [0,1]
                              Resize(input_size),
                              SubtractMeans(color_mean) #Subtract mean của BGR
                              ]),

            "val": Compose([ConvertFromInts(), #convert imf from int to float32
                            Resize(input_size),
                            SubtractMeans(color_mean)
            ])
        }


    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == '__main__':
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    #prepare train, valid, annotation list
    root_path = "data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    idx = 1
    img_file_path = val_img_list[idx]

    #read img
    img = cv2.imread(img_file_path)
    height, width, channel = img.shape

    #annotation information
    trans_anno = Anno_xml(classes)
    anno_info_list = trans_anno(val_annotation_list[idx], width, height)

    #plot original img
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

