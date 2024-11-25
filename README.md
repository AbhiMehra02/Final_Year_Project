Rice Seeds Classification using AI/ML

Project Overview
This project uses machine learning techniques to classify five types of rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag, using image data. The dataset contains 75,000 images (15,000 per variety) sourced from Kaggle.

Models Used
Xception: Best performing model with fine-tuning, achieving 99.92% accuracy.
VGG19: Shows good results, but fine-tuning the whole model reduces performance.
InceptionResNetV2: High accuracy, with 99.9% achieved.
Key Features
Transfer Learning: Using pre-trained models (Xception, VGG19, InceptionResNetV2) for feature extraction.
Fine-tuning: Adjusting the last layers for better performance.
Data Augmentation: Applied to enhance the model’s robustness and prevent overfitting.
Best Practices & Results
Data Augmentation: Increases accuracy by adding transformations like rotation, flipping, and scaling.
Ensemble Learning: Combining predictions from multiple models improves overall accuracy.
Future Work
Investigating more data augmentation techniques.
Exploring lightweight models for deployment in production.
Installation
Clone the repository:

bash```
git clone https://github.com/yourusername/rice-agriculture-ml.git
```
Install dependencies:

bash```
pip install -r requirements.txt
```

Run the Streamlit app:
bash```
streamlit run app.py
```
