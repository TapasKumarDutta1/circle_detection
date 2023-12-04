# Circle Detection

The repository conatins code for circle detection from noise images. 
A [Colab Notebook](https://github.com/TapasKumarDutta1/circle_detection/blob/main/demo.ipynb) is provided for utilizing different files and experimenting with the anomaly detection process.
The project uses shufflenet and is trained from scratch using adam optimizer and Mean Squared Error loss and iou(0.5 and 0.5-0.95) threshold accuracy as evaluation metric.

## Technical Details
1. Architecture: The project leverages the Shufflenet architecture, a powerful neural network design optimized for efficiency and accuracy in circle detection.

2. Training Strategy: The model is trained from scratch using the Adam optimizer and Mean Squared Error loss function. Evaluation metrics include Intersection over Union (IoU) with thresholds set at 0.5 and a range from 0.5 to 0.95. To ensure robust training, the project incorporates dynamic strategies such as "reduce lr on plateau" and "early stopping" to prevent overfitting.

3. Data Splits: The dataset is strategically divided into 10,000 samples for training, 1,000 for validation, and 2,000 for testing, providing a comprehensive evaluation of the model's performance across diverse datasets.


## Exploratory Attempts
Two strategies were tested and found to be less effective:

1. Pretrained Models: Initial experiments with pretrained models(Imagenet) were conducted, but it was concluded that a training-from-scratch approach better addressed the unique challenges inherent in circle detection from noisy images.

2. Augmentation Techniques: Despite efforts to enhance performance through data augmentation, creating more samples within the training process was ultimately more successful in achieving the desired results.

## Sample output plotted against training samples
![images](img/sample_output.png)

## Note
Since the work extensively uses cv2.circle for visualization the row and col variables are reversed during dataset creation, model prediction and evaluation.
