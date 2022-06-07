from torchvision import transforms
from utils.data.iCIFAR100 import iCIFAR100


datasets_all = {
    'CIFAR100': iCIFAR100,
}

AVAILABLE_TRANSFORMS_train = {
    'CIFAR100': transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
}

AVAILABLE_TRANSFORMS_test = {
    'CIFAR100': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
}
