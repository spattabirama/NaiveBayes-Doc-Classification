# Movie Review Classification with Multinomial Naive Bayes

This project focuses on training a Multinomial Naive Bayes algorithm to classify movie reviews as positive or negative using the IMDB Large Movie Review dataset. The algorithm utilizes a Bag-of-Words representation for documents and employs probability estimation based on training set examples.

## Key Components

1. **Multinomial Naive Bayes Algorithm:**
   - Represents documents as Bag-of-Words vectors.
   - Computes prior probabilities (Pr(yi)) and conditional probabilities (Pr(wk | yi)) for training.
   - Classifies new documents based on maximizing the probability equation.

2. **IMDB Dataset:**
   - Contains 25,000 positive and 25,000 negative movie reviews for training and testing.
   - Utilizes 'train' and 'test' folders with 'pos' and 'neg' subfolders.

3. **Data Preprocessing:**
   - Converts reviews to lowercase, removes punctuation and stopwords.

4. **Experiments and Analyses:**
   - Investigates the impact of classifying instances using both posterior probabilities and log-probabilities.
   - Implements Laplace Smoothing with varying α values and analyzes their impact.
   - Explores the effect of training set size on model performance.
   - Studies model behavior with an imbalanced dataset.

## Results

- Log probabilities significantly improve recall and overall accuracy compared to posterior probabilities.
- Optimal Laplace Smoothing (α=1) enhances accuracy, precision, and recall.
- Minimal impact observed on performance with varying training set sizes (50% vs. 100%).
- Imbalanced datasets lead to biased predictions, impacting accuracy and recall.

## Detailed Documentation

- Find detailed experiment results, performance metrics, and visualizations in the provided documentation.
