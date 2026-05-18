from .cifar import (
    CIFAR3,
    CIFAR3_Noise,
    CIFAR10,
    CIFAR10_minimal,
    CIFAR10_Noise,
    CIFAR100,
    CIFAR100_Noise,
    CIFAR100_LT,
    CIFAR10_LT,
)
from .mnist import (
    MNIST,
    MNIST_Noise,
    FashionMNIST,
    FashionMNIST_Noise,
)
from .tinyimagenet import (
    TinyImageNet,
    TinyImageNet_Noise,
)
from .makeblobs import (
    MakeBlobs,
    MakeBlobs_Noise,
)
from .teacher_generated import (
    Teacher_Generated,
    Teacher_Generated_Noise,
)
from .pacs import PACS