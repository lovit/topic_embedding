from .math import train_svd

def graph_to_svd_embedding(X, n_components):
    """
    It transform graph to embedding space using SVD

    :param X: numpy.ndarray or scipy.spare.matrix
        Representation matrix. Shape = (n_items, n_features)
    :param n_components: int
        Dimension of embedding space

    It returns
    ----------
    wv : numpy.ndarray
        Embedding space
    mapper : numpy.ndarray
        Embedding mapper
        wv = X * mapper
    """

    U, Sigma, VT = train_svd(X, n_components=100)
    wv = U * (Sigma ** (1/2))
    mapper = VT.T * (Sigma ** (-1/2))
    return wv, mapper