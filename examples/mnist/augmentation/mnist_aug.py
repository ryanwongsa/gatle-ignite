from torchvision import transforms

from examples.mnist.dataloaders.mnist_dataset import MEAN, STD


class Transformation:
    """A torchvision Compose in a class: the contract asks only for a callable."""

    def __init__(self, degrees=10, translate=0.1):
        self.transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=degrees, translate=(translate, translate)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
