from Classification.config import *
from image_transform import ImageTransform
from utils import make_datapath_list, train_model, params_to_update
from my_dataset import MyDataSet


def main():
    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')

    #dataset
    train_dataset = MyDataSet(train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = MyDataSet(val_list, transform=ImageTransform(size, mean, std), phase='val')

    #dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    #network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # loss
    criterior = nn.CrossEntropyLoss()

    # optimizer
    params1, params2, params3 = params_to_update(net)

    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4},
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3},
    ], momentum=0.9)

    #training
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)


if __name__ == '__main__':
    main()
