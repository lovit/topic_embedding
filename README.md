
## Usage

Using 2016-10-20 news,

```python
x # scipy.sparse.csr_matrix. (doc, term) frequency matrix
idx_to_vocab # list of str
vocab_to_idx # dict. str -> int
```

Transforming doc-term frequency matrix to term-term co-occurrence graph (weight is relative occurrence proportion score)

```python
from soytopic import tf_to_prop_graph

g_prop, g_count = tf_to_prop_graph(x)
```

Embedding graph to topic space

```python
from soytopic import graph_to_svd_embedding

wv, mapper = graph_to_svd_embedding(g_prop, n_components=300)
```

For searching topically similar words with `경제`

```python
from soytopic import most_similar

most_similar('경제', wv, vocab_to_idx, idx_to_vocab, topk=10)
```

```
[('불확실성', 0.6302248361139866),
 ('국내총생산', 0.6189045253328803),
 ('구조개혁', 0.5911689592265104),
 ('경제성장', 0.5782112455093333),
 ('수출', 0.5666036470892614),
 ('성장', 0.5512108035597899),
 ('지속', 0.5452329420410507),
 ('30년', 0.541947651720095),
 ('재정', 0.5408913175786476),
 ('경제성장률', 0.5382736375167195)]
```

For searching topically similar words with `아이오아이`

```python
most_similar('아이오아이', wv, vocab_to_idx, idx_to_vocab, topk=10)
```

```
[('엠카운트다운', 0.9753215812259417),
 ('엠넷', 0.943666545230428),
 ('타이틀곡', 0.9323954166872636),
 ('파워풀', 0.9118913739918781),
 ('안무', 0.9090141395793365),
 ('싱글', 0.9059985769284604),
 ('가요계', 0.9007329902290891),
 ('래핑', 0.8984225424143106),
 ('소녀들', 0.8944131123670225),
 ('멤버들', 0.8892673837539027)]
```

For searching topically similar words with `최순실`

```python
most_similar('최순실', wv, vocab_to_idx, idx_to_vocab, topk=10)
```

```
[('게이트', 0.9333657722243184),
 ('실세', 0.8898127720942869),
 ('스포츠재단', 0.8890545574731947),
 ('최순실씨', 0.8850366691670046),
 ('모녀', 0.8812997481546445),
 ('최씨', 0.8675594389448809),
 ('비선', 0.8653632407037474),
 ('비선실세', 0.8607325527171852),
 ('미르', 0.8472184338555011),
 ('의혹', 0.8459249142649861)]
```