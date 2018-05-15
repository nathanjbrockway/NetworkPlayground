import random
import sys
from mnist import MNIST

sys.path.append('../')

mndata = MNIST("../data")

images, labels = mndata.load_testing()

index = random.randrange(0,len(images))
print(mndata.display(images[index]))
           