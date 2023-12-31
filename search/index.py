import math
import numpy

from numpy.typing import ArrayLike
from search import embedding
from sklearn.cluster import KMeans
from tenseal.tensors.ckksvector import CKKSVector


class Index:
    """
    Index class for efficient searching in a corpus using clustering and matrix representation.

    Parameters:
    - model (embedding.Model): The embedding model for encoding text.
    - corpus (list[str]): List of texts to build the index upon.

    Attributes:
    - corpus (ArrayLike): Numerical vectors representing the encoded corpus.
    - clusters (int): The number of clusters formed during initialization.
    - centroids (ArrayLike): The centroids of clusters.
    - matching (list[tuple[int, ArrayLike]]): Pairs of cluster labels and corresponding embeddings.
    - matrix (ArrayLike): Matrix representation of the clustered embeddings.

    Methods:
    - _clusterize() -> None: Private method to clusterize the corpus using KMeans.
    - _index() -> None: Private method to create a matrix index from the clustered embeddings.
    - search(query: CKKSVector) -> ArrayLike: Search the index for the closest match to the input query.

    Example:
    ```python
    model = embedding.Model("bert-base-nli-mean-tokens")
    corpus = ["text1", "text2", "text3"]
    index = Index(model, corpus)
    query_vector = model.encode("search query")
    result = index.search(query_vector)
    ```
    """

    def __init__(self, model: embedding.Model, corpus: list[str]):
        self.corpus = model.encode(corpus)
        self.embedding_size = len(self.corpus[0])
        self._clusterize()
        self._index()

    def _clusterize(self) -> None:
        """
        Clusterize the corpus using KMeans algorithm.

        Returns:
        - None
        """
        n_clusters = math.ceil(math.sqrt(len(self.corpus)))
        clustering = KMeans(n_clusters=n_clusters, n_init="auto").fit(self.corpus)

        self.clusters = n_clusters
        self.centroids = clustering.cluster_centers_
        self.matching = list(zip(clustering.labels_, self.corpus))

    def _index(self) -> None:
        """
        Create a matrix index from the clustered embeddings.

        Returns:
        - None
        """
        index = [[] for _ in range(self.clusters)]

        for cluster, embedding in self.matching:
            index[cluster].append(embedding)

        filler = numpy.ones(self.embedding_size) * -10_000
        self.max_size = max([len(val) for val in index])

        for cluster, embedding in enumerate(index):
            cluster_size = len(embedding)
            index[cluster].extend([filler] * (self.max_size - cluster_size))

        self.matrix = numpy.array(index)

    def search(self, query: CKKSVector) -> ArrayLike:
        """
        Search the index for the closest match to the input query.

        Parameters:
        - query (CKKSVector): The encrypted query vector.

        Returns:
        - ArrayLike: Result vector representing the closest match.
        """
        matrix = self.matrix.reshape(self.clusters, self.max_size * self.embedding_size)

        return query @ matrix
