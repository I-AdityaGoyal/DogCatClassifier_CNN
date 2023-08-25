# 🐶🐱 DogCat Classifier using CNN 🚀


Welcome to the DogCat Classifier repository! This project is all about classifying adorable dog and cat images using Convolutional Neural Networks (CNNs) 🖼️. The model is trained on a fantastic dataset of 20,000 images sourced from Kaggle. Not only that, we've even deployed it using Streamlit for easy interaction.

## 📦 Dataset

The dataset used for this project can be found on Kaggle: [DogCat Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## 🧠 Model Architecture

The classifier harnesses the power of a CNN architecture, and we've taken it up a notch by employing the pre-trained VGG16 model through transfer learning for perfectly accurate predictions!

## ⚙️ Setup

Follow these steps to get the ball rolling:

1. **Clone the repository:**
 ```bash
 git clone https://github.com/I-AdityaGoyal/DogCatClassifier_CNN.git
 ```

2. **Navigate to the project directory:**
```bash
cd DogCatClassifier_CNN
```


3. **Install Dependencies:**
Make sure you're all set with the libraries by running:
```bash
pip install -r requirements.txt
```



4. **Download and Prepare the Dataset:**
Fetch the dataset from the provided Kaggle link and cozy it up in a directory named `dataset`. Don't forget to have separate folders for `train` and `test` data.

5. **Model Training and Evaluation:**
Execute the Jupyter Notebook or Python script that takes care of training and evaluation. We'll be fine-tuning the VGG16 model and training our classifier on the dataset.

6. **Making Predictions:**
Once the model is trained, you can predict on new images using the trained weights. Modify the prediction script to suit your needs.


## 🚀 Streamlit Deployment

Guess what? We've made it even easier to interact with our model! We've deployed it using Streamlit. Just use the `model.pickle` file and run the `app.py` script to fire up the app and start classifying images right away!

```bash
streamlit run app.py
```
## 🚀 Usage and Customization

Feel free to add your personal touch by tweaking the code – play around with various CNN architectures, hyperparameters, and image augmentation techniques. Explore the plethora of pre-trained models in TensorFlow/Keras for your transfer learning adventures.

## 🤝 Contributions

Contributions are more than welcome! If you sniff out any issues or have cool enhancements in mind, open an issue or wag a pull request – we're all ears.

---
Get ready to unleash some cuteness overload! 🐾
