# Private Web Search POC

This Proof of Concept (POC) replicates the functionality described in the article "Private Web Search with Tiptoe" by Alexandra Henzinger et al. The POC focuses on building an index for efficient searching in a corpus using clustering and matrix representation, with the added privacy-preserving feature using the Tiptoe approach.

## Overview

The code consists of three main components:

1. **Model Class (`Model`):**
   - Utilizes the SentenceTransformer library for text embedding.
   - Enables encoding of input text or a list of texts into numerical vectors.

2. **Index Class (`Index`):**
   - Takes a corpus of texts and utilizes the `Model` for encoding.
   - Clusterizes the corpus using the KMeans algorithm.
   - Creates a matrix index from the clustered embeddings.
   - Provides a method for searching the index for the closest match to an encrypted query vector.

3. **Client Class (`Client`):**
   - Enables interaction with the clustering model using encrypted queries.
   - Utilizes the `Model` for encoding and the `Index` for efficient searching.

## Usage Example

```python
dataset = ["text1", "text2", "text3"]

query = "text1"

# Initialize the encryption context using CKKS scheme with specific parameters
context = tenseal.context(
    tenseal.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.generate_galois_keys()
context.global_scale = 2**40

# Instantiate a Model
model = Model(id="paraphrase-MiniLM-L6-v2")

# Instantiate an Index with the Model and the provided dataset
index = Index(model=model, corpus=dataset)

# Instantiate a Client for interacting with the clustering model using encrypted queries
client = Client(model=model, centroids=index.centroids, context=context)

# Create an encrypted query vector based on the input text
query = client.query(text)

# Decrypt the result obtained from searching the Index with the encrypted query
result = client.decrypt(index.search(query))

# Rank the documents based on the result vector and the original text
client.rank(result=result, text=text)
```

## Installation

This project is managed using Poetry. To install the project, you can use the following commands:


```bash
pip install poetry
poetry install
```

These commands ensure that Poetry is installed, and the project dependencies are properly set up.

## Notes

- This POC draws inspiration from the "Private Web Search with Tiptoe" article, concentrating on replicating the fundamental functionality outlined in the paper.
- The original paper employs distinct clustering and encryption methods that contribute to better scalability.
- In this POC, the focus lies on ranking documents, with the retrieval of the actual text from the server being omitted.
- All operations are performed locally. While it is conceivable to expand this approach by segregating the index onto a server, such an extension falls beyond the scope of the current project.

## Citation for the Original Authors

```bibtex
@inproceedings{Henzinger2023,
  doi = {10.1145/3600006.3613134},
  url = {https://doi.org/10.1145/3600006.3613134},
  year = {2023},
  month = {October},
  publisher = {ACM},
  author = {Alexandra Henzinger and Emma Dauterman and Henry Corrigan-Gibbs and Nickolai Zeldovich},
  title = {Private Web Search with Tiptoe},
  booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles}
}
```

**REPO URL:** https://github.com/ahenzinger/tiptoe
