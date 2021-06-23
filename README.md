# Object Centric Image Detection


<!-- ABOUT THE PROJECT -->
### About The Project

Implemented pretrained deep learning models (InceptionV3, DenseNet121, ResNet50) which integrates methods like data augmentation, autoencoder and transfer learning to boost performance. 

Then, classify the images using CUDA-based Machine Learning algorithm (ELM) for parallel-processing and achieved a 96% accuracy on cifar10 dataset.

### Built With

Tools and framework used in the project.
* [Python](https://python.org/)
* [CUDA C](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Nvidia Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [TensorFlow](https://www.tensorflow.org/)
* [keras](https://keras.io/)
* [numpy](https://numpy.org/)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Sklearn](https://scikit-learn.org/stable/)


<!-- GETTING STARTED -->
## Getting Started

Step by step guide of how project works


### Dataset
[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* 60000 images (50000 for training and 10000 for testing)
* classes: 10

[Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)
* 60000 images (50000 for training and 10000 for testing)
* classes: 100

[Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
* 9144 images (3,060 for training and 6,084 for testing)
* classes: 101

[Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
* 30607 images (10100 for training and 20506 for testing)
* classes: 256

**This Project particulary working on cifar10 and cifar100 dataset. However, I've tried all the dataset that are given above.**

### Installation

1. Get a datasets from above links
2. Clone the repo
   ```sh
   git clone https://github.com/parthvalani/Extreme-Learning-Machine.git
   ```
3. Install python packages for feature extraction
   ```sh
   pip install keras sklearn numpy OpenCV
   ```
4. Download Nvidia CUDA Toolkit (CUDA 11.1) for GPU utilization and install it properly in TensorFlow envionment. 
5. Open cmd and write following command
  ```sh
   nvcc cifar10.cu -lcurand -lcublas -o cifar10.exe
   ```
6. Run 'cifar10.exe' in cuda_elm folder to see output.


<!-- USAGE EXAMPLES -->
## Methodology

* It starts with python. 
* Load respective dataset
* Preprocess the data (normalizing, one-hot encoding)
* Divide in train-test set
* Built a pretrained on imagenet 'Densenet101' with transfer learning
* Extract features from that model through generating new model with last layer as extracted layer
* Again, Scalling the inputs and transpose it for the cuda elm
* Save it in 'cuda_elm' folder as csv file
* Try with python based ELM using sklearn
* Now, built [Extereme Learning Machine](https://towardsdatascience.com/introduction-to-extreme-learning-machines-c020020ff82b) (ELM) Algorthm in C with any IDEs
* Transfer the extract csv file from main memary to GPU memory using cuda C
* Perform elm algorithm in cuda with GPU memory
* Pass the result to CPU memory and print the accuracy

## Conclusion
After trying python-based ELM. It's been conlcuded that parellel utilization of the resourses (GPU) gives better accuracy and faster results. 
* Time for predication
  - Python ELM: 100s
  - Cuda ELM: 2s


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact
LinkiedIn - [parthvalani](https://www.linkedin.com/in/parthvalani/) - parthnvalani@gmail.com

Project Link: [https://github.com/parthvalani/Extreme-Learning-Machine](https://github.com/parthvalani/Extreme-Learning-Machineg)
