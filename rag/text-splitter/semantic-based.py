"""
- for now this technique is ok ok
- may be in future it will more stable
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

text = """
For many years, sequence modelling and generation was done by using plain recurrent neural networks (RNNs). A well-cited early example was the Elman network (1990). In theory, the information from one token can propagate arbitrarily far down the sequence, but in practice the vanishing-gradient problem leaves the model's state at the end of a long sentence without precise, extractable information about preceding tokens.

A key breakthrough was LSTM (1995),[note 1] an RNN which used various innovations to overcome the vanishing gradient problem, allowing efficient learning of long-sequence modelling. One key innovation was the use of an attention mechanism which used neurons that multiply the outputs of other neurons, so-called multiplicative units.[13] Neural networks using multiplicative units were later called sigma-pi networks[14] or higher-order networks.[15] LSTM became the standard architecture for long sequence modelling until the 2017 publication of transformers. However, LSTM still used sequential processing, like most other RNNs.[note 2] Specifically, RNNs operate one token at a time from first to last; they cannot operate in parallel over all tokens in a sequence.


Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.[33] Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.



Modern transformers overcome this problem, but unlike RNNs, they require computation time that is quadratic in the size of the context window. The linearly scaling fast weight controller (1992) learns to compute a weight matrix for further processing depending on the input.[16] One of its two networks has "fast weights" or "dynamic links" (1981).[17][18][19] A slow neural network learns by gradient descent to generate keys and values for computing the weight changes of the fast neural network which computes answers to queries.[16] This was later shown to be equivalent to the unnormalized linear transformer.[20][21]

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language. Python 3.0, released in 2008, was a major revision and not completely backward-compatible with earlier versions. Beginning with Python 3.5,[34] capabilities and keywords for typing were added to the language, allowing optional static typing.[35] As of 2026, the Python Software Foundation supports Python 3.10, 3.11, 3.12, 3.13, and 3.14, following the project's annual release cycle and five-year support policy. Python 3.15 is currently in the alpha development phase, and the stable release is expected to come out in October 2026."[36]Earlier versions in the 3.x series have reached end-of-life and no longer receive security updates.

"""
# text = "My name is utsav \n\n he lived in navsari"
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

location = Path("/").cwd() / "learn-langchain" / "data.pdf"
splitter = SemanticChunker(
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

res = splitter.create_documents(texts=[text])


print(f"{len(res)=}")

print(f"{res=}")
