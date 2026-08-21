import numpy as np


def create_embedding(token_id: int, dimensions: int = 8) -> np.ndarray:
    """
    Create a deterministic toy embedding for a token.

    This is not an actual LLM embedding.
    It is used to understand how token IDs
    can be represented as vectors.
    """
    rng = np.random.default_rng(seed=token_id)

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=dimensions,
    ).astype(np.float32)


def build_embedding_matrix(
    token_ids: list[int],
    dimensions: int = 8,
) -> np.ndarray:
    """Create one embedding vector for each token ID."""
    vectors = [
        create_embedding(token_id, dimensions)
        for token_id in token_ids
    ]

    return np.stack(vectors)


if __name__ == "__main__":
    token_ids = [101, 205, 307, 412]

    matrix = build_embedding_matrix(token_ids)

    print("Token IDs:", token_ids)
    print("Embedding matrix shape:", matrix.shape)
    print("First embedding:", matrix[0])