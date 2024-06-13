from libs.tools_ollama import launch_server_ollama
import ollama

print("TEST 1: launch ollama...")
launch_server_ollama(mdl="llama3", temperature=0, top_p=1, name="llama3-coherent")
response = ollama.chat(model='llama3-coherent', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
#response = ollama.generate(model='llama3-coherent', prompt='Why is the sky blue?')
print(response)
print("TEST 1: launch ollama done.")

print("TEST 2 : launch ollama config default...")
launch_server_ollama()
response = ollama.chat(model='llama3-coherent', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
print(response)
print("TEST 2 : launch ollama config default done.")

print("TEST 3 : launch ollama config main...")
launch_server_ollama(config="main")
response = ollama.chat(model='llama3-coherent', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
print(response)
print("TEST 3 : launch ollama config main done.")
