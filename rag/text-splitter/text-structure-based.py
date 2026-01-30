from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
For many years, sequence modelling and generation was done by using plain recurrent neural networks (RNNs). A well-cited early example was the Elman network (1990). In theory, the information from one token can propagate arbitrarily far down the sequence, but in practice the vanishing-gradient problem leaves the model's state at the end of a long sentence without precise, extractable information about preceding tokens.

A key breakthrough was LSTM (1995),[note 1] an RNN which used various innovations to overcome the vanishing gradient problem, allowing efficient learning of long-sequence modelling. One key innovation was the use of an attention mechanism which used neurons that multiply the outputs of other neurons, so-called multiplicative units.[13] Neural networks using multiplicative units were later called sigma-pi networks[14] or higher-order networks.[15] LSTM became the standard architecture for long sequence modelling until the 2017 publication of transformers. However, LSTM still used sequential processing, like most other RNNs.[note 2] Specifically, RNNs operate one token at a time from first to last; they cannot operate in parallel over all tokens in a sequence.

Modern transformers overcome this problem, but unlike RNNs, they require computation time that is quadratic in the size of the context window. The linearly scaling fast weight controller (1992) learns to compute a weight matrix for further processing depending on the input.[16] One of its two networks has "fast weights" or "dynamic links" (1981).[17][18][19] A slow neural network learns by gradient descent to generate keys and values for computing the weight changes of the fast neural network which computes answers to queries.[16] This was later shown to be equivalent to the unnormalized linear transformer.[20][21]
"""
# text = "My name is utsav \n\n he lived in navsari"
from pathlib import Path

location = Path("/").cwd() / "learn-langchain" / "data.pdf"
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=2)

res = splitter.split_text(text=text)

# loader = PyPDFLoader(file_path=location)
# res = splitter.split_documents(loader.load())

print(f"{len(res)=}")
print(f"{res[1]=}")
print(f"{res=}")
