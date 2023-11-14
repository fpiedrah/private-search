import tenseal
import numpy
from numpy.typing import ArrayLike
from search import embedding
from tenseal.enc_context import Context
from tenseal.tensors.ckksvector import CKKSVector

class Client:
    """
    Client class for interacting with a clustering model using encrypted queries.

    Parameters:
    - model (embedding.Model): The clustering model.
    - centroids (ArrayLike): The centroids of clusters.
    - context (Context): The encryption context.

    Attributes:
    - model (embedding.Model): The clustering model.
    - centroids (ArrayLike): The centroids of clusters.
    - clusters (int): The number of clusters.
    - size (int): The size of the centroids.
    - context (Context): The encryption context.

    Methods:
    - _distance(embedding: str) -> ArrayLike: Private method to compute distances between centroids and an embedding.
    - query(text: str) -> CKKSVector: Generate an encrypted query vector based on the input text.
    - decrypt(result: CKKSVector) -> ArrayLike: Decrypt the result vector obtained from the server.
    - rank(result: ArrayLike, text: str) -> list[int]: Rank the clusters based on the result vector and input text.
    """
    def __init__(self, model: embedding.Model, centroids: ArrayLike, context: Context):
        self.model = model
        self.centroids = centroids
        self.clusters = len(centroids)
        self.size = self.centroids.shape[1]
        self.context = context

    def _distance(self, embedding: str) -> ArrayLike:
        """
        Compute the Euclidean distance between centroids and an embedding.

        Parameters:
        - embedding (str): The embedding to calculate distances from.

        Returns:
        - ArrayLike: Array of distances between centroids and the input embedding.
        """
        distance = (self.centroids - embedding) ** 2
        distance = numpy.sum(distance, axis=1)
        distance = numpy.sqrt(distance)

        return distance

    def query(self, text: str) -> CKKSVector:
        """
        Generate an encrypted query vector based on the input text.

        Parameters:
        - text (str): The input text for the query.

        Returns:
        - CKKSVector: The encrypted query vector.
        """
        query = numpy.zeros(self.clusters)
        index = numpy.argmin(self._distance(self.model.encode(text)))

        query[index] = 1

        return tenseal.ckks_vector(self.context, query)

    def decrypt(self, result: CKKSVector) -> ArrayLike:
        """
        Decrypt the result vector obtained from the server.

        Parameters:
        - result (CKKSVector): The encrypted result vector.

        Returns:
        - ArrayLike: The decrypted result as a NumPy array.
        """
        result = numpy.array(result.decrypt())

        return result.reshape(len(result) // self.size, self.size)

    def rank(self, result: ArrayLike, text: str) -> list[int]:
        """
        Rank the clusters based on the result vector and input text.

        Parameters:
        - result (ArrayLike): The result vector.
        - text (str): The input text for ranking.

        Returns:
        - list[int]: The ranked list of clusters.
        """
        ranking = result @ self.model.encode(text).T

        return numpy.flip(numpy.argsort(ranking)).tolist()
