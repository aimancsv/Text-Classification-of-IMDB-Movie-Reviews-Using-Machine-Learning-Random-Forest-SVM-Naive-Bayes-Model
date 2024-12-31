This repo explores sentiment analysis and text classification using natural language processing (NLP) and machine learning techniques. It combines foundational NLP methods like tokenization, stemming, and POS tagging with supervised models, including Random Forest, Naive Bayes, and SVM, to classify sentiment in the IMDB movie review dataset. Through exploratory data analysis, feature engineering, and hyperparameter optimization, SVM emerged as the most effective model. This comprehensive pipeline demonstrates the power of combining NLP and machine learning for real-world text classification tasks, with future potential for deep learning and multilingual analysis.

**Supervised Text Classification and Model Evaluation**:


### Key Components

1. **Exploratory Data Analysis (EDA)**:
   - **Dataset Selection**:
     - Utilized the **IMDB Dataset of 50K Movie Reviews**, a large corpus for binary sentiment classification (positive/negative).
     - <img width="500" alt="Screenshot 2024-12-31 at 8 16 09 AM" src="https://github.com/user-attachments/assets/30f75a43-8b73-4182-863c-9b57a198a09e" />

   - **Data Cleaning and Preprocessing**:
     - Addressed missing and duplicate values.
     - Preprocessed text data by:
       - Lowercasing text.
       - Removing HTML tags, URLs, emojis, and stop words.
       - Tokenizing and lemmatizing words for cleaner analysis.
       - <img width="500" alt="Screenshot 2024-12-31 at 8 19 03 AM" src="https://github.com/user-attachments/assets/2929dcaf-fdd3-4cbf-a3c6-0e3e5b3d1fee" />
       - <img width="500" alt="Screenshot 2024-12-31 at 8 20 15 AM" src="https://github.com/user-attachments/assets/0fc7d6c1-6b76-4a21-9a91-6f42c75535c5" />


   - **Visualization**:
     - Word frequency distributions.
     - <img width="500" alt="Screenshot 2024-12-31 at 8 22 57 AM" src="https://github.com/user-attachments/assets/6d5bb606-1da8-494b-8e4a-ed09971459ee" />

     - Sentiment polarity histograms.
     - <img width="500" alt="Screenshot 2024-12-31 at 8 23 20 AM" src="https://github.com/user-attachments/assets/476c19fa-0dc8-461b-bb97-6b1381f3d4f8" />
     - <img width="500" alt="Screenshot 2024-12-31 at 8 23 58 AM" src="https://github.com/user-attachments/assets/64626307-396b-422c-9a30-d2f21e1522cb" />

     - Word clouds for positive and negative sentiment words.
     - <img width="500" alt="Screenshot 2024-12-31 at 8 25 23 AM" src="https://github.com/user-attachments/assets/b990f3e0-d34e-4686-981d-e0da2407746f" />
     - <img width="500" alt="Screenshot 2024-12-31 at 8 26 13 AM" src="https://github.com/user-attachments/assets/3ea96457-1eb0-4c6d-a966-3581e0206bb0" />






2. **Supervised Learning Models**:
   - **Model Training and Evaluation**:
     - Three machine learning models were implemented:
       1. **Random Forest**: Robust ensemble-based model for handling complex relationships.
          <img width="500" alt="Screenshot 2024-12-31 at 8 26 45 AM" src="https://github.com/user-attachments/assets/d14b7fb6-9571-487f-900e-8f2a6ce2b818" />
          <img width="500" alt="Screenshot 2024-12-31 at 8 28 54 AM" src="https://github.com/user-attachments/assets/d604f50e-0da5-4e00-9f99-182c5584a759" />


       2. **Naive Bayes**: A fast, probabilistic model ideal for text-based classification.
          <img width="500" alt="Screenshot 2024-12-31 at 8 28 05 AM" src="https://github.com/user-attachments/assets/177950f0-7f2a-4663-9534-037115dd2df3" />
          <img width="500" alt="Screenshot 2024-12-31 at 8 29 09 AM" src="https://github.com/user-attachments/assets/c24ba5ad-caf1-4d1b-b116-74a67ef22a36" />


       3. **Support Vector Machine (SVM)**: A powerful model for separating data with a clear margin in high-dimensional spaces.
          <img width="500" alt="Screenshot 2024-12-31 at 8 28 22 AM" src="https://github.com/user-attachments/assets/083046f0-9bf3-4e52-9351-b7ae13311072" />
          <img width="500" alt="Screenshot 2024-12-31 at 8 29 27 AM" src="https://github.com/user-attachments/assets/c42007b7-cf36-476d-86b6-2480af00c5bf" />


     - Models were trained using TF-IDF vectorized text data to extract features and assign weights based on term frequency and document importance.
   - **Hyperparameter Optimization**:
     - Conducted grid search to optimize model parameters for better accuracy and performance.
     - Parameters such as the number of estimators (Random Forest), alpha values (Naive Bayes), and kernel functions (SVM) were fine-tuned.
     - **Random Forest:**
     - <img width="1157" alt="Screenshot 2024-12-31 at 8 40 37 AM" src="https://github.com/user-attachments/assets/eec574d5-beb3-4345-a59c-19ec4a10e844" />

     - **Naive Bayes:**
     - <img width="500" alt="Screenshot 2024-12-31 at 8 36 07 AM" src="https://github.com/user-attachments/assets/34c516fe-f6b9-4fb6-bdf4-c25215d6eda1" />
     - **Support Vector Machine (SVM):**
     - <img width="500" alt="Screenshot 2024-12-31 at 8 37 41 AM" src="https://github.com/user-attachments/assets/f6eeb8b2-6932-43b4-ac91-bd7000dc87ff" />



     


3. **SAS Text Miner Integration**:
   - **Automated Text Classification**:
     - Leveraged **SAS Text Miner** to extract linguistic features and build classification rules.
     - Implemented data partitioning, text parsing, and text filtering for enhanced accuracy.
   - **Rule-Based Models**:
     - Generated human-readable rules to interpret sentiment patterns.
     - Analyzed model fit statistics for validation and error rates.
     - <img width="500" alt="Screenshot 2024-12-31 at 8 30 48 AM" src="https://github.com/user-attachments/assets/6188951f-e9c7-45d3-bca7-a31734547863" />
     - <img width="500" alt="Screenshot 2024-12-31 at 8 31 33 AM" src="https://github.com/user-attachments/assets/8611c9fd-5d0e-4823-8e29-a74d2fd91156" />
     - <img width="500" alt="Screenshot 2024-12-31 at 8 31 51 AM" src="https://github.com/user-attachments/assets/b0e8f5ab-6b2d-4b77-ab37-119bbd66751c" />




4. **Evaluation and Metrics**:
   - **Performance Metrics**:
     - Accuracy, precision, recall, and F1-score were calculated for each model.
     - Confusion matrices were used to visualize model performance across sentiment categories.
   - **Model Comparisons**:
     - **Random Forest**: High accuracy and robust handling of noisy data.
     - **Naive Bayes**: Computationally efficient with good performance for high-dimensional text.
     - **SVM**: Best precision and recall, excelling in binary classification tasks.
   - **Results**:
     - **Random Forest:** Accuracy 87%, Precision 85%, Recall 86%.
     - **Naive Bayes:** Accuracy 86%, Precision 85%, Recall 89%.
     - **SVM:** Accuracy 88%, Precision 89%, Recall 87%.

### Value Proposition
This module showcases the practical application of supervised learning for sentiment analysis, illustrating how various models can be trained, optimized, and evaluated to handle real-world text classification tasks. The inclusion of SAS Text Miner demonstrates the integration of automated tools into a machine learning pipeline.

### Use Cases
- Sentiment analysis for customer reviews and social media posts.
- Automated text classification in email filtering and topic modeling.
- Developing predictive models for business intelligence and market analysis.

### Future Directions
- Extend the model to multi-class classification (e.g., neutral sentiment).
- Incorporate advanced deep learning models like LSTM or BERT for improved performance.
- Experiment with multilingual datasets for global applicability.

---
