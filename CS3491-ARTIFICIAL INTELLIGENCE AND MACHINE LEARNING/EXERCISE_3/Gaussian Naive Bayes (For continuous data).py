import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        # Calculate means, variances and priors for each class
        self.classes = np.unique(y)
        self.mean = {}
        self.variance = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def _gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.variance[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])  # log(P(c))
            class_conditional = np.sum(np.log(self._gaussian_pdf(c, x)))  # log(P(x|c))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]


# Example usage:
# Create some sample data for 2 classes
X = np.array([[1.0, 2.1], [1.1, 1.9], [3.1, 2.9], [3.0, 3.2], [4.0, 4.5], [5.0, 5.0]])
y = np.array([0, 0, 1, 1, 1, 1])

gnb = GaussianNaiveBayes()
gnb.fit(X, y)
predictions = gnb.predict(X)
print("Predictions:", predictions)
