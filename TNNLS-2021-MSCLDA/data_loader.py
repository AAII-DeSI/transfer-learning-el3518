from torchvision import datasets, transforms
import torch
from PIL import Image

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
    
## Below are for ImageCLEF datasets

class ImageCLEF(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, transform=None):
        super(ImageCLEF, self).__init__()
        self.transform = transform
        file_name = root_path + 'list/' + dir + 'List.txt'
        lines = open(file_name, 'r').readlines()
        self.images, self.labels = [], []
        self.dir = dir
        for item in lines:
            line = item.strip().split(' ')
            self.images.append(root_path + dir + '/' + line[0].split('/')[-1])
            self.labels.append(int(line[1].strip()))

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(img)
        return image, target

    def __len__(self):
        return len(self.images)

def load_imageclef_train(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageCLEF(root_path=root_path, dir=dir, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_imageclef_test(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageCLEF(root_path=root_path, dir=dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
