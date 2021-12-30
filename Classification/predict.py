from lib import *
from config import *
from utils import load_model
from image_transform import ImageTransform


class_index = ["ants", "bees"]


class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, output):
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_index[0]

        return predicted_label


predictor = Predictor(class_index)

def predict(img):
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.eval()

    #prepare model
    model = load_model(net, save_path)

    #prepare input img
    transform = ImageTransform(resize=size, mean=mean, std=std)
    img = transform(img=img, phase="test")

    img = img.unsqueeze_(0) #(chanel, height, weight) -> (1, chanel, height, weight)

    #predict
    output = model(img)
    response = predictor.predict_max(output)
    
    return response


if __name__ == '__main__':
    img = Image.open('ants.jpg')
    pred = predict(img)
    print(pred)
