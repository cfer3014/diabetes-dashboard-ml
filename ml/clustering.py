from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def build_clusters(X_scaled, df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return kmeans, pca, X_pca, df