from langchain_core.runnables import RunnableParallel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    # model="gemini-3-flash-preview",
    model="gemini-2.5-flash-lite",
)
# propmt = PromptTemplate(
#     template="Generate 5 facts about {topic}", input_variables=["topic"]
# )

# chain = propmt | model
# res = chain.invoke({"topic": "javascript"})
# print(res.content[0])


report = """
    In machine learning, a neural network (NN) or neural net, also called an artificial neural network (ANN), is a computational model inspired by the structure and functions of biological neural networks.[1][2]

A neural network consists of connected units or nodes called artificial neurons, which loosely model the neurons in the brain. Artificial neuron models that mimic biological neurons more closely have also been recently investigated and shown to significantly improve performance. These are connected by edges, which model the synapses in the brain. Each artificial neuron receives signals from connected neurons, then processes them and sends a signal to other connected neurons. The "signal" is a real number, and the output of each neuron is computed by some non-linear function of the totality of its inputs, called the activation function. The strength of the signal at each connection is determined by a weight, which adjusts during the learning process.

Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer) to the last layer (the output layer), possibly passing through multiple intermediate layers (hidden layers). A network is typically called a deep neural network if it has at least two hidden layers.[3]

Artificial neural networks are used for various tasks, including predictive modeling, adaptive control, and solving problems in artificial intelligence. They can learn from experience, and can derive conclusions from a complex and seemingly unrelated set of information.
"""
prompt1 = PromptTemplate(
    template="Generate points wise notes for '{text}'", input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 MCQ based on following text '{text}' ",
    input_variables=["text"],
)


# WAY - 1
# chain1 = prompt1 | model | StrOutputParser()
# chain2 = prompt2 | model | StrOutputParser()
# full_chain = RunnableParallel(notes=chain1, quiz=chain2)

# res = full_chain.invoke({"text": report})

# print(res)

# WAY - 2
prompt3 = PromptTemplate(
    template="merge the notes and quiz in to single document notes->'{notes}'  quiz->'{quiz}'",
    input_variables=["notes", "quiz"],
)

paraller_chain = RunnableParallel(
    notes=prompt1 | model | StrOutputParser(),
    quiz=prompt2 | model | StrOutputParser(),
)

merge_chain = prompt3 | model | StrOutputParser()

full_chain = paraller_chain | merge_chain
res = full_chain.invoke({"text": report})
print(res)
full_chain.get_graph().print_ascii()
