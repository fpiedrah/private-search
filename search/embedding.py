from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer
from typing import Union


class Model:
    """
    Wrapper class for a SentenceTransformer model to simplify text encoding.

    Parameters:
    - id (str): Identifier for the SentenceTransformer model to be used.

    Attributes:
    - _model (SentenceTransformer): The underlying SentenceTransformer model.

    Methods:
    - encode(data: Union[list[str], str]) -> ArrayLike: Encode input text or a list of texts into numerical vectors.

    Example:
    ```python
    model = Model("bert-base-nli-mean-tokens")
    encoded_data = model.encode("Hello, how are you?")
    ```
    """

    def __init__(self, id: str):
        self._model = SentenceTransformer(id)

    def encode(self, data: Union[list[str], str]) -> ArrayLike:
        """
        Encode input text or a list of texts into numerical vectors.

        Parameters:
        - data (Union[list[str], str]): Input text or list of texts to be encoded.

        Returns:
        - ArrayLike: Numerical vectors representing the encoded input data.
        """
        return self._model.encode(data)
