import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import dok_matrix
from scipy.sparse import vstack
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import safe_sparse_dot


def tf_to_prop_graph(X, min_score=0.75, min_cooccurrence=1,
    verbose=True, include_self=True, topk=-1):
    """
    :param X: scipy.sparse
        Shape = (n_docs, n_terms)
    :param min_score: float
        Minimum co-occurrence score; pos_prop / (pos_prop + neg_prop)
        Default is 0.75
    :param min_cooccurrence: int
        Mininum co-occurrence count. Default is 1
    :param verbose: Boolean
        If True, it shows progress status for each 100 words
    :param include_self: Boolean
        If True, (w0, w0) is included in graph with score 1.0.

    It returns
    ----------
    g_prop : scipy.sparse.csr_matrix
        Co-occurrence score matrix. Shape = (n_terms, n_terms)
    g_count : scipy.sparse.csr_matrix
        Co-occurrence count matrix. Shape = (n_terms, n_terms)
    """

    to_count = lambda X:np.asarray(X.sum(axis=0)).reshape(-1)
    to_prop = lambda X: X / X.sum()

    total_count = to_count(X)
    n_vocabs = X.shape[1]

    rows = []
    cols = []
    props = []
    counts = []

    for base_idx in range(n_vocabs):
        pos_docs = X[:,base_idx].nonzero()[0]
        pos_count = to_count(X[pos_docs])
        ref_count = total_count - pos_count
        pos_prop = to_prop(pos_count)
        ref_prop = to_prop(ref_count)
        prop = pos_prop / (pos_prop + ref_prop)

        if min_cooccurrence > 1:
            idx_mc = np.where(pos_count >= min_cooccurrence)[0]
            idx_ms = np.where(prop >= min_score)[0]
            rel_idxs = np.intersect1d(idx_mc, idx_ms)
        else:
            rel_idxs = np.where(prop >= min_score)[0]

        for idx in rel_idxs:
            if not include_self and idx == base_idx:
                continue
            rows.append(base_idx)
            cols.append(idx)
            props.append(prop[idx])
            counts.append(pos_count[idx])

        if verbose and base_idx % 100 == 0:
            print('\rcreate graph %d / %d ...' % (base_idx , n_vocabs), end='')

    if verbose:
        print('\rcreate graph from %d words was done' % n_vocabs)

    g_prop = csr_matrix((props, (rows, cols)), shape=(n_vocabs, n_vocabs))
    g_count = csr_matrix((counts, (rows, cols)), shape=(n_vocabs, n_vocabs))

    return g_prop, g_count

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
            C = larger_than(C, min_count)
    else:
        stacks = []
        n_batch = math.ceil(n_terms / batch_size)
        for i in range(n_batch):
            b = i * batch_size
            e = min(n_terms, (i+1) * batch_size)
            C = safe_sparse_dot(XT[b:e], X)
            if min_count > 1:
                C = larger_than(C, min_count)
            stacks.append(C)
        C = vstack(stacks)

    if not isinstance(C, csr_matrix):
        C = C.tocsr()

    return C

def larger_than(X, min_count):
    rows_, cols_ = X.nonzero()
    data_ = X.data
    rows, cols, data = [], [], []
    for r, c, d in zip(rows_, cols_, data_):
        if d < min_count:
            continue
        rows.append(r)
        cols.append(c)
        data.append(d)
    return csr_matrix((data, (rows, cols)), shape=X.shape)

def _as_diag(px, alpha):
    if len(px.shape) == 1:
        px_diag = diags(px.tolist())
    else:
        px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag

def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    indices = np.where(exp_pmi.data < min_exp_pmi)[0]
    exp_pmi.data[indices] = 1

    # apply logarithm
    exp_pmi.data = np.log(exp_pmi.data)
    return exp_pmi

def train_pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    """
    :param X: scipy.sparse.csr_matrix
        (word, contexts) sparse matrix
    :param py: numpy.ndarray
        (1, word) shape, probability of context words.
    :param min_pmi: float
        Minimum value of pmi. all the values that smaller than min_pmi
        are reset to zero.
        Default is zero.
    :param alpha: float
        Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
        Default is 0.0
    :param beta: float
        Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) )
        Default is 1.0
    It returns
    ----------
    pmi : scipy.sparse.dok_matrix or scipy.sparse.csr_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of rows (items)
    py : numpy.ndarray
        Probability of columns (features)
    """

    # convert x to probability matrix & marginal probability
    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py ** beta
        py /= py.sum()
    pxy = X / X.sum()

    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py

def train_svd(X, n_components, n_iter=5, random_state=None):
    """
    :param X: scipy.sparse.csr_matrix
        Input matrix
    :param n_components: int
        Size of embedding dimension
    :param n_iter: int
        Maximum number of iteration. Default is 5
    :param random_state: random state
        Default is None
    It returns
    ----------
    U : numpy.ndarray
        Representation matrix of rows. shape = (n_rows, n_components)
    Sigma : numpy.ndarray
        Eigenvalue of dimension. shape = (n_components, n_components)
        Diagonal value are in decreasing order
    VT : numpy.ndarray
        Representation matrix of columns. shape = (n_components, n_cols)
    """

    if (random_state == None) or isinstance(random_state, int):
        random_state = check_random_state(random_state)

    n_features = X.shape[1]

    if n_components >= n_features:
        raise ValueError("n_components must be < n_features;"
                         " got %d >= %d" % (n_components, n_features))

    U, Sigma, VT = randomized_svd(
        X, n_components,
        n_iter = n_iter,
        random_state = random_state)

    return U, Sigma, VT