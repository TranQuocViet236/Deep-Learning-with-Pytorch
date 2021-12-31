from libs import *
from make_datapath import make_datapath_list


class Anno_xml(object):

    def __init__(self, classes):
        self.classes = classes


    def __call__(self, xml_path, width, height):
        #include image annotation
        ret = []

        #read xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)

            if difficult == 1:
                continue

            #information for bouding box
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in pts:
                pixel = int(bbox.find(pt).text) -1

                if pt == 'xmin' or pt == 'xmax':
                    pixel /= width #ratio of width

                else:
                    pixel /= height #ratio of height

                bndbox.append(pixel)

            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += [bndbox]

        return np.array(ret) #[[xmin, ymin, xmax, ymax, label_id]...]


if __name__ == '__main__':
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    anno_xml = Anno_xml(classes)

    root_path = "data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    idx = 1
    img_file_path = val_img_list[idx]

    img = cv2.imread(img_file_path) #(height, weight, channel: BGR)
    height, width, channel = img.shape #get size img

    # print(f"Size img: ({height}, {width}, {channel})")
    annotation_infor = anno_xml(val_annotation_list[idx], width, height)

    print(annotation_infor)




