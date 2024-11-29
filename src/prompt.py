general_medrag_system = """You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Your responses will be used for research purposes only, so please have a definite answer.\n\n"""

pubmedqa_medrag = """
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}

Answer:"""

general_medrag = """
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}
D. {option_4}

Answer:"""

# RAGなしのprompt
general_pure = """
Here is the question:
{question}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}
D. {option_4}

Answer:"""

pubmedqa_pure = """
Here is the question:
{question}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}

Answer:"""

general_medrag_pattern1 = """
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}
D. {option_4}

Answer:"""

general_medrag_pattern2 = """
Here is the question:
{question}

Here are the relevant documents:
{context}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}
D. {option_4}

Answer:"""

general_medrag_pattern3 = """
Here is the question:
{question}

Here are the potential choices:
A. {option_1}
B. {option_2}
C. {option_3}
D. {option_4}

Here are the relevant documents:
{context}

Answer:"""
