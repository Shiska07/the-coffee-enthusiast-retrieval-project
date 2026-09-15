""" Prompt templates for the Coffee RAG system. """

from langchain_classic.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# ---------------- System prompt for the generation stage. This is the "persona" of the main generator LLM. -----

GENERATOR_SYSTEM_PROMPT = """

You are a knowledgeable coffee expert answering questions \
using ONLY the coffee review excerpts provided as context.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly \
rather than guessing.
- When relevant, mention which roaster(s) or coffee(s) or review url to support your answer.
- Keep your answer concise and directly responsive to the question.

"""

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


# ------------------------------- system prompt for hallucination grader ------------------------------

HL_GRADER_SYSTEM_PROMPT = """

"You check whether an answer is grounded in the provided context. It counts "
    "as supported ONLY if every factual claim can be verified from the context "
    "alone; plausible detail not in the context counts as NOT supported."
    
"""

HL_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HL_GRADER_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
    ]
)


# SYSTEM PROMPTS FOR QUERY TRANSLATION  

# -------------------------  multi-query translation system prompt ---------------------------

MQ_TRANSLATION_SYSTEM_PROMPT = """
You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}
"""

MQ_TRANSLATION_PROMPT = ChatPromptTemplate.from_template(
    MQ_TRANSLATION_SYSTEM_PROMPT
)


# --------------------------- reciprocal rank-fusion system prompt ----------------------------------

RRF_TRANSLATION_SYSTEM_PROMPT = """
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):
"""

RRF_TRANSLATION_PROMPT = ChatPromptTemplate.from_template(
    RRF_TRANSLATION_SYSTEM_PROMPT
)


# ---------------------------------- query-decomposition system prompt ---------------------------------

QD_TRANSLATION_SYSTEM_PROMPT_P1 = """
You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):
"""

QD_TRANSLATION_PROMPT_P1 = ChatPromptTemplate.from_template(
    QD_TRANSLATION_SYSTEM_PROMPT_P1
)

QD_TRANSLATION_SYSTEM_PROMPT_P2 = """
Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

QD_TRANSLATION_PROMPT_P2 = ChatPromptTemplate.from_template(
    QD_TRANSLATION_SYSTEM_PROMPT_P2
)


# -------------------- step-back query translation system prompt ---------------------------------

SB_TRANSLATION_SYSTEM_PROMPT = """
You are a knowledgeable coffee expert. Your task is to step back and paraphrase a question to a more 
generic step-back question, which is easier to answer. Here are a few examples:
"""

SB_QA_EXAMPLES = [
    {
        "input": "I love sweet, vanilla, dessert-like coffee notes. What should I look for?",
        "output": "Which coffees are described as having sweet, vanilla, or dessert-like tasting notes?",
    },
    {
        "input": "I try to only buy certified organic and fair trade coffee. What are some highly-rated options?",
        "output": "Which certified organic and fair-trade coffees are highly rated?",
    },
    {
        "input": "Can you suggest a chocolatey, rich espresso blend that's medium-light roast?",
        "output": "What medium-light roast espresso blends are described as chocolatey and rich?",
    },
    {
        "input": "What are the single highest-rated, most acclaimed coffees in your database?",
        "output": "What are the highest-rated, most acclaimed coffees overall?",
    },
    {
        "input": "I love floral, jasmine-like light roast coffees from Ethiopia — almost tea-like. What should I try?",
        "output": "Which Ethiopian light roast coffees have floral, jasmine-like, or tea-like tasting notes?",
    },
]

SB_QA_EXAMPLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

FEW_SHOT_PROMPT = FewShotChatMessagePromptTemplate(
    example_prompt=SB_QA_EXAMPLE_PROMPT,
    examples=SB_QA_EXAMPLES
)

SB_TRANSLATION_SYSTEM_PROMPT = ChatPromptTemplate.from_messages(
    ("system", SB_TRANSLATION_SYSTEM_PROMPT),
    FEW_SHOT_PROMPT,
    ("human", "{question}")
)

