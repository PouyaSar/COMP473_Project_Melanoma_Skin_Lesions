
Required Installs for all Imports:

pip install scikit-image
pip install opencv-python
pip install numpy
pip install pandas
pip install scikit-learn
pip install imbalanced-learn
pip install joblib
pip install torch

User Manual:

    Extraction of a single lesion:
        Go into the file named "main.py". 
        Change the value of filename to the name of the image you want to extract the features of and also change the path to the path where the image(s) are stored. (e.g: filename = "ISIC_0035995.jpg" and img_path = f"HAM10000/{filename}").
        Then run the file. Extracts the features to the lesion_features.csv file.

    Batch Extraction and model creation:
        Go into the file named "batch.py".
        Change the path to the path where the images are located. Since HAM10000 uses a sequentional numbered naming scheme we made it iterate starting from the first image "ISIC_0024306.jpg" and it increaes by 10000 to "ISIC_003406.jpg" and iterates through all those images and extracts the features into the lesion_features_batch.csv file.
        To run the model go into the file named "classifier.py".
        Now run this file. This will create the learning model based of the data set in the lesion_features_batch.csv.

    Predict whether a lesion is malignant or benign:
        Go into the file named "predict.py".
        This will use the features of the lesion in lesion_features.csv and the existing model to predict whether the lesion is benign or malignant. 

