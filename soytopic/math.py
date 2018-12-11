from math import ceil
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import vstack
from sklearn.utils.extmath import safe_sparse_dot

def tf_to_cooccurrence(X, min_count=1, batch_size=-1):
    """
    :param X: scipy.sparse
        Shape = (n_docs, n_terms)
    :param min_count: int
        Mininum co-occurrence count. Default is 1
    :param batch_size: int
        The number of words in a batch. Default is 2000

    It returns
    ----------
    C : scipy.sparse.csr_matrix
        Co-occurrence matrix
    """

    XT = X.T
    n_terms = X.shape[1]
    if batch_size == -1:
        C = safe_sparse_dot(XT, X)
        if min_count > 1:
            C = C > min_count
    else:
        stacks = []
        n_batch = ceil(n_terms / batch_size)
        for i in range(n_batch):
            b = i * batch_size
            e = min(n_terms, (i+1) * batch_size)
            C = safe_sparse_dot(XT[b:e], X)
            if min_count > 1:
                C = C > min_count
            stacks.append(C)
        C = vstack(stacks)

    if not isinstance(C, csr_matrix):
        C = C.tocsr()

    return C