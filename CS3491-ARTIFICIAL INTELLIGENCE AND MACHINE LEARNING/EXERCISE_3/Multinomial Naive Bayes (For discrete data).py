import numpy as np

class MultinomialNaiveBayes:
    def fit(self, X, y):
        # Calculate priors and likelihoods for each class
        self.classes = np.unique(y)
        self.class_count = len(self.classes)
        self.feature_count = X.shape[1]
        
        # Likelihood (P(word|class)) and priors P(class)
        self.likelihood = np.zeros((self.class_count, self.feature_count))
        self.priors = np.zeros(self.class_count)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.likelihood[idx, :] = np.sum(X_c, axis=0) + 1  # Laplace smoothing
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])
        
        # Normalize likelihood to represent probabilities
        self.likelihood = self.likelihood / np.sum(self.likelihood, axis=1, keepdims=True)
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_likelihood = np.sum(np.log(self.likelihood[idx, :]) * x)
            posterior = prior + class_likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]


# Example usage with text data
# Columns are words in a vocabulary, rows are document word counts
X = np.array([[3, 2, 0], [1, 1, 0], [0, 0, 5], [0, 1, 4]])
y = np.array([0, 0, 1, 1])  # Two classes (e.g., spam vs non-spam)

mnb = MultinomialNaiveBayes()
mnb.fit(X, y)
predictions = mnb.predict(X)
print("Predictions:", predictions)
