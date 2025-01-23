# Week1
README: CNN Image Classification with Data Visualization
Overview
This project uses Convolutional Neural Networks (CNN) for image classification. The code loads and processes images from the specified training and testing directories, creates a Pandas DataFrame, and visualizes the category distribution with a pie chart.

Prerequisites
Install the required libraries:

bash
Copy
pip install tensorflow opencv-python matplotlib pandas tqdm
Directory Structure
Ensure your dataset is structured as:

markdown
Copy
DATASET/
    TRAIN/
        Category1/
        Category2/
    TEST/
        Category1/
        Category2/
Code Explanation
Data Loading: Images are read from the TRAIN directory, converted to RGB, and stored in x_data (images) and y_data (labels).

Data Visualization: A pie chart displays the distribution of categories based on the label column of the DataFrame.

Model Placeholder: After data processing, you can create and train a CNN model (not shown in the code).

How to Use
Organize your dataset into TRAIN and TEST directories.
Run the script to load images and view the distribution pie chart.
Build and train a CNN model with the processed data.
Example CNN model (for reference):

python
Copy
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
Conclusion
This code processes images, visualizes category distributions, and prepares the data for a CNN-based classification task.


