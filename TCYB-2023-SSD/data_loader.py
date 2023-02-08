from torchvision import datasets, transforms
import torch
from PIL import Image

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()#,transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
         ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()#,transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
         ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

##dataset with image list
class ImageComS(torch.utils.data.Dataset):
    def __init__(self, root_path, domains, task, classlist, sel_sam, transform=None):
        super(ImageComS, self).__init__()
        self.transform = transform
        self.images, self.labels = [], []#i=0
        for i in range(len(task)-1):
            file_name = root_path + domains[task[i]] + sel_sam + '.txt'
            lines = open(file_name, 'r').readlines() 
            #images, labels =[],[]# item[last-1]
            for item in lines:
                for classname in classlist:
                    last=item.rfind('/')
                    lasts=item.rfind('/',0,last-1)
                    #flag = classname in item
                    if classname==item[lasts+1:last]:
                        line = item.strip().split(' ')
                        self.images.append(root_path + domains[task[i]] + '/' + classname + '/'  + line[0].split('/')[-1])
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

def load_image_ComS(root_path, domains, task, classlist, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageComS(root_path=root_path, domains=domains, task=task, classlist=classlist,sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


############################################################################
class ImageTSS(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, classlist, sel_sam, transform=None):
        super(ImageTSS, self).__init__()
        self.transform = transform
        file_name = root_path + dir + sel_sam + '.txt'
        lines = open(file_name, 'r').readlines()
        ##################################
        #dsize = len(lines)
        #tr_size = int(0.9*dsize)
        #_, te_txt = torch.utils.data.random_split(lines, [tr_size, dsize - tr_size])
        ######################################
        
        self.images, self.labels = [], []
        self.dir = dir
        for item in lines:
            for classname in classlist:
                last=item.rfind('/')
                lasts=item.rfind('/',0,last-1)
                #flag = classname in item
                if classname==item[lasts+1:last]:
                    line = item.strip().split(' ')
                    self.images.append(root_path + dir + '/' + classname + '/' + line[0].split('/')[-1])
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
############################################
class ImageTSS_tra(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, classlist, sel_sam, rate, transform=None):
        super(ImageTSS_tra, self).__init__()
        self.transform = transform
        file_name = root_path + dir + sel_sam + '.txt'
        lines = open(file_name, 'r').readlines()
        ##################################
        dsize = len(lines)
        tr_size = int(rate*dsize)
        tr_txt, _ = torch.utils.data.random_split(lines, [tr_size, dsize - tr_size])
        ######################################
        
        self.images, self.labels = [], []
        self.dir = dir
        for item in tr_txt:
            for classname in classlist:
                last=item.rfind('/')
                lasts=item.rfind('/',0,last-1)
                #flag = classname in item
                if classname==item[lasts+1:last]:
                    line = item.strip().split(' ')
                    self.images.append(root_path + dir + '/' + classname + '/' + line[0].split('/')[-1])
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

class ImageTSS_val(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, classlist, sel_sam, rate, transform=None):
        super(ImageTSS_val, self).__init__()
        self.transform = transform
        file_name = root_path + dir + sel_sam + '.txt'
        lines = open(file_name, 'r').readlines()
        ##################################
        dsize = len(lines)
        tr_size = int(rate*dsize)
        _, te_txt = torch.utils.data.random_split(lines, [tr_size, dsize - tr_size])
        ######################################
        
        self.images, self.labels = [], []
        self.dir = dir
        for item in te_txt:
            for classname in classlist:
                last=item.rfind('/')
                lasts=item.rfind('/',0,last-1)
                #flag = classname in item
                if classname==item[lasts+1:last]:
                    line = item.strip().split(' ')
                    self.images.append(root_path + dir + '/' + classname + '/' + line[0].split('/')[-1])
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

def load_image_TSS_val(root_path, dir, classlist, sel_sam, rate, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    
    train_loader = {}
    data = ImageTSS_tra(root_path=root_path, dir=dir, classlist=classlist,sel_sam=sel_sam, rate = rate, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader['tra'] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    data = ImageTSS_val(root_path=root_path, dir=dir, classlist=classlist,sel_sam=sel_sam, rate = rate, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader['val'] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    
    return train_loader
#########################################################
class ImageTSS_idx(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, classlist, sel_sam, transform=None):
        super(ImageTSS_idx, self).__init__()
        self.transform = transform
        file_name = root_path + dir + sel_sam + '.txt'
        lines = open(file_name, 'r').readlines()

        self.images, self.labels = [], []
        self.dir = dir
        for item in lines:
            for classname in classlist:
                last=item.rfind('/')
                lasts=item.rfind('/',0,last-1)
                #flag = classname in item
                if classname==item[lasts+1:last]:
                    line = item.strip().split(' ')
                    self.images.append(root_path + dir + '/' + classname + '/' + line[0].split('/')[-1])
                    self.labels.append(int(line[1].strip()))
    
    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(img)
        return image, target, index

    def __len__(self):
        return len(self.images)

def load_image_TSS(root_path, dir, classlist, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    
    data = ImageTSS(root_path=root_path, dir=dir, classlist=classlist,sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

    
def load_image_TSS_test(root_path, dir, classlist, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageTSS(root_path=root_path, dir=dir, classlist=classlist,sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader  

def load_image_TSS_select(root_path, dir, classlist, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageTSS(root_path=root_path, dir=dir, classlist=classlist,sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader  

def load_image_TSS_test_idx(root_path, dir, classlist, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageTSS_idx(root_path=root_path, dir=dir, classlist=classlist, sel_sam=sel_sam, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

         
class Imagelist(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, classlist, transform=None):
        super(Imagelist, self).__init__()
        self.transform = transform
        file_name = root_path + dir + 'List.txt'
        lines = open(file_name, 'r').readlines()
        self.images, self.labels = [], []
        self.dir = dir
        for item in lines:
            for classname in classlist:
                last=item.rfind('/')
                lasts=item.rfind('/',0,last-1)
                #flag = classname in item
                if classname==item[lasts+1:last]:
                    line = item.strip().split(' ')
                    self.images.append(root_path + dir + '/' + classname + '/' + line[0].split('/')[-1])
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
        
class Imagelist_idx(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, classlist, transform=None):
        super(Imagelist_idx, self).__init__()
        self.transform = transform
        file_name = root_path + dir + 'List.txt'
        lines = open(file_name, 'r').readlines()
        self.images, self.labels = [], []
        self.dir = dir
        for item in lines:
            for classname in classlist:
                last=item.rfind('/')
                lasts=item.rfind('/',0,last-1)
                #flag = classname in item
                if classname==item[lasts+1:last]:
                    line = item.strip().split(' ')
                    self.images.append(root_path + dir + '/' + classname + '/' + line[0].split('/')[-1])
                    self.labels.append(int(line[1].strip()))
    
    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(img)
        return image, target, index

    def __len__(self):
        return len(self.images)

def load_image_train(root_path, dir, classlist, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = Imagelist(root_path=root_path, dir=dir, classlist=classlist, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_image_test(root_path, dir, classlist, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = Imagelist(root_path=root_path, dir=dir, classlist=classlist, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

def load_image_select(root_path, dir, classlist, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = Imagelist(root_path=root_path, dir=dir, classlist=classlist, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader

def load_image_test_idx(root_path, dir, classlist, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = Imagelist_idx(root_path=root_path, dir=dir, classlist=classlist, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

class CLEFComS(torch.utils.data.Dataset):
    #def __init__(self, root_path, dir, transform=None):
    def __init__(self, root_path, domains, task, sel_sam, transform=None):
        super(CLEFComS, self).__init__()
        self.transform = transform
        self.images, self.labels = [], []#i=0
        for i in range(len(task)-1): #+ 'list/' 
            file_name = root_path + domains[task[i]] + sel_sam + '.txt'
            lines = open(file_name, 'r').readlines() 
            #images, labels =[],[]# item[last-1]
            for item in lines:
                line = item.strip().split(' ')
                self.images.append(root_path + domains[task[i]] + '/'  + line[0].split('/')[-1])
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

def load_CLEF_ComS(root_path, domains, task, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = CLEFComS(root_path=root_path, domains=domains, task=task, sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader   

class CLEFTSS(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, sel_sam, transform=None):
        super(CLEFTSS, self).__init__()
        self.transform = transform # + 'list/' 
        file_name = root_path + dir + sel_sam + '.txt'
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

def load_CLEF_TSS(root_path, dir, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = CLEFTSS(root_path=root_path, dir=dir, sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_CLEF_TSS_test(root_path, dir, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = CLEFTSS(root_path=root_path, dir=dir, sel_sam=sel_sam, transform=transform)
#    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader
###########################################
class CLEFTSS_tra(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, sel_sam, transform=None):
        super(CLEFTSS_tra, self).__init__()
        self.transform = transform # + 'list/' 
        file_name = root_path + dir + sel_sam + '.txt'
        lines = open(file_name, 'r').readlines()
        ##################################
        dsize = len(lines)
        tr_size = int(0.6*dsize)
        tr_txt, _ = torch.utils.data.random_split(lines, [tr_size, dsize - tr_size])
        ######################################
        
        self.images, self.labels = [], []
        self.dir = dir
        for item in tr_txt:
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
class CLEFTSS_val(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, sel_sam, transform=None):
        super(CLEFTSS_val, self).__init__()
        self.transform = transform # + 'list/' 
        file_name = root_path + dir + sel_sam + '.txt'
        lines = open(file_name, 'r').readlines()
        ##################################
        dsize = len(lines)
        tr_size = int(0.6*dsize)
        _, te_txt = torch.utils.data.random_split(lines, [tr_size, dsize - tr_size])
        ######################################
        
        self.images, self.labels = [], []
        self.dir = dir
        for item in te_txt:
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

def load_CLEF_TSS_val(root_path, dir, sel_sam, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    train_loader = {}
    data = CLEFTSS_tra(root_path=root_path, dir=dir, sel_sam=sel_sam, transform=transform)
    train_loader['tra'] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    data = CLEFTSS_val(root_path=root_path, dir=dir, sel_sam=sel_sam, transform=transform)
    train_loader['val'] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader    
## Below are for ImageCLEF datasets

class ImageCLEF(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, transform=None):
        super(ImageCLEF, self).__init__()
        self.transform = transform # + 'list/' 
        file_name = root_path + dir + 'List.txt'
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

def load_imageclef_select(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageCLEF(root_path=root_path, dir=dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader

class ImageCLEF_idx(torch.utils.data.Dataset):
    def __init__(self, root_path, dir, transform=None):
        super(ImageCLEF_idx, self).__init__()
        self.transform = transform # + 'list/' 
        file_name = root_path + dir + 'List.txt'
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
        return image, target, index

    def __len__(self):
        return len(self.images)
def load_imageclef_test_idx(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
         #transforms.RandomCrop(224),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageCLEF_idx(root_path=root_path, dir=dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
