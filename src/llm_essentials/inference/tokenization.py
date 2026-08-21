import tiktoken


TOKENIZER_NAME = "cl100k_base"

tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)


def tokenize(text: str) -> list[int]:
    """Convert text into token IDs."""
    return tokenizer.encode(text)


def show_tokens(text: str) -> None:
    """Display token IDs and their decoded representations."""
    token_ids = tokenize(text)

    print(f"Text: {text}")
    print(f"Token count: {len(token_ids)}")

    for position, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id])
        print(f"{position}: {token_id} -> {token_text!r}")


if __name__ == "__main__":
    sample_text = "Large language models learn patterns from text."
    show_tokens(sample_text)