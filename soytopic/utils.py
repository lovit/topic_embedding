from sklearn.metrics import pairwise_distances


def most_similar(query, wv, vocab_to_idx, idx_to_vocab, topk=10):
    """
    :param query: str
        String type query word
    :param wv: numpy.ndarray or scipy.sparse.matrix
        Topical representation matrix
    :param vocab_to_idx: dict
        Mapper from str type query to int type index
    :param idx_to_vocab: list
        Mapper from int type index to str type words
    :param topk: int
        Maximum number of similar terms.
        If set top as negative value, it returns similarity with all words

    It returns
    ----------
    similars : list of tuple
        List contains tuples (word, cosine similarity)
        Its length is topk
    """

    q = vocab_to_idx.get(query, -1)
    if q == -1:
        return []
    qvec = wv[q].reshape(1,-1)
    dist = pairwise_distances(qvec, wv, metric='cosine')[0]
    sim_idxs = dist.argsort()
    if topk > 0:
        sim_idxs = sim_idxs[:topk+1]
    similars = [(idx_to_vocab[idx], 1 - dist[idx]) for idx in sim_idxs if idx != q]
    return similars