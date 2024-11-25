Rice Seeds Classification using AI/ML

Project Overview
This project uses machine learning techniques to classify five types of rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag, using image data. The dataset contains 75,000 images (15,000 per variety) sourced from Kaggle.

## Models Used

- **Xception**: Best performing model with fine-tuning, achieving **99.92% accuracy**.
- **VGG19**: Shows good results, but fine-tuning the entire model reduces performance.
- **InceptionResNetV2**: Achieves high accuracy, with **99.9%**.

---

## Key Features

- **Transfer Learning**: Leveraging pre-trained models (Xception, VGG19, InceptionResNetV2) for feature extraction.
- **Fine-tuning**: Adjusting the last layers of models for better performance.
- **Data Augmentation**: Enhancing robustness and preventing overfitting by applying transformations.

---

## Best Practices & Results

- **Data Augmentation**: Significantly increases accuracy through transformations like rotation, flipping, and scaling.
- **Ensemble Learning**: Combining predictions from multiple models to improve overall accuracy.

---

## Future Work

- Investigating additional data augmentation techniques for further improvements.
- Exploring lightweight models for efficient deployment in production environments.


Installation
Clone the repository:

bash
```
git clone https://github.com/yourusername/rice-agriculture-ml.git
```
Install dependencies:

bash
```
pip install -r requirements.txt
```

Run the Streamlit app:
bash
```
streamlit run app.py
```
