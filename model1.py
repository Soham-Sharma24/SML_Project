import cv2
import skimage
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA

# Parameters for LBP
radius = 1  # How far to look around each pixel
n_points = 8 * radius  # Number of points to consider

def extract_lbp_features(img, radius=1, n_points=8):
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist

def extract_hog_features(img):
    features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
    return features

# Process each image and extract LBP or HOG features
def extract_features(data_dir, method='lbp', apply_pca=False, pca_components=100):
    emotion_folders = os.listdir(data_dir)
    feature_list = []
    labels = []
    
    for emotion in emotion_folders:
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                if method == 'lbp':
                    features = extract_lbp_features(img, radius, n_points)
                elif method == 'hog':
                    features = extract_hog_features(img)
                feature_list.append(features)
                labels.append(emotion)

    features = np.array(feature_list)
    
    # Optionally apply PCA
    if apply_pca:
        n_components = min(pca_components, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(features)
    
    return features, labels

# Example: Extract LBP features from the training set and apply PCA
train_features, train_labels = extract_features(train_dir, method='lbp', apply_pca=True, pca_components=50)

# Display the size of the extracted features
print(f"Extracted features shape: {train_features.shape}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Split the extracted features into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_train), yticklabels=set(y_train))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
