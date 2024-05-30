'''
Test on Langchain : libertai llama-cpp online model free access


'''
# import
from libs.libertai import Libertai

# definitions

 # TEST with LLAMA 3 8B: Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf
llm = Libertai(
    api_base="https://curated.aleph.cloud/vm/84df52ac4466d121ef3bb409bb14f315de7be4ce600e8948d71df6485aa5bcc3",
    temperature=0,
    top_p=1,
    max_length=800
)


'''llm = Libertai(
    temperature=0,
    top_p=1,
    max_length=800
)'''

question_1 = "Quels sont les besoins dans le domaine des pratiques RH que la blockchain adresse ou pourrait adresser ?"
# REPONSE TEST :
print("\n\n TEST 1 : REPONSE LLM only : \n")
print("Question 1 :", question_1)
results_1 = llm.invoke(question_1)
print("Response 1 : ")
print(results_1)

print("END")
