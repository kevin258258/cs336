import json
from functools import lru_cache
from pathlib import Path


_FIXTURE_ROOT = Path(__file__).resolve().parent / "tests" / "fixtures"
_VOCAB_PATH = _FIXTURE_ROOT / "gpt2_vocab.json"
_ENDOFTEXT = "<|endoftext|>"
_GPT2_PAT_STR = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)


@lru_cache(maxsize=1)
def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs)))


@lru_cache(maxsize=1)
def _gpt2_constructor_from_fixtures() -> dict[str, object]:
    byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
    with _VOCAB_PATH.open() as vocab_f:
        gpt2_vocab = json.load(vocab_f)

    mergeable_ranks = {
        bytes(byte_decoder[ch] for ch in token): rank
        for token, rank in gpt2_vocab.items()
        if token != _ENDOFTEXT
    }
    return {
        "name": "gpt2",
        "explicit_n_vocab": len(gpt2_vocab),
        "pat_str": _GPT2_PAT_STR,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {_ENDOFTEXT: gpt2_vocab[_ENDOFTEXT]},
    }


def _install_tiktoken_fixture_patch() -> None:
    try:
        import tiktoken.registry as registry
        import tiktoken_ext.openai_public as openai_public
    except Exception:
        return

    openai_public.ENCODING_CONSTRUCTORS["gpt2"] = _gpt2_constructor_from_fixtures
    registry.ENCODINGS.pop("gpt2", None)
    if registry.ENCODING_CONSTRUCTORS is not None:
        registry.ENCODING_CONSTRUCTORS["gpt2"] = _gpt2_constructor_from_fixtures


_install_tiktoken_fixture_patch()
