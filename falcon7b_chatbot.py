import torch 
from transformers import pipeline 



#initiate the model
Local_Model = "tiiuae/falcon-7b-instruct"

generator = pipeline(model=Local_Model, device = -1, torch_dtype = torch.bfloat16, pad_token_id=11)


chat_history = [{"role": "system", "content":"You are an assistant named AI."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit","exit","bye"]:
        print("AI: Goodbye! Thank you for using my services!")
        exit()
    
    #add your own input to chat history 
    chat_history.append({"role": "user", "content": user_input})

    #format the full prompt from history 
    formatted_prompt = ""
    for message in chat_history:
        role = message['role'].capitalize()
        formatted_prompt += f"{role}: {message['content']}\n"
    formatted_prompt += "AI: "

    #generate the response
    generation = generator(
        formatted_prompt,
        do_sample=False,
        temperature=1.0,
        top_p=1,
        max_new_tokens=60
    )
        
    # response from the model
    dirty_response = generation[0]['generated_text']

    # Remove the prompt from the generated text
    if dirty_response.startswith(formatted_prompt):
        response = dirty_response[len(formatted_prompt):]
    else:
        response = dirty_response

    # Stop at the first new "User:" or "You:" or "AI:" to prevent the model from continuing the convo
    for stop_word in ["User:", "You:", "AI:", "\nUser:", "\nYou:", "\nAI:"]:
        if stop_word in response:
            response = response.split(stop_word)[0]


    #add AI response to history 
    chat_history.append({"role":"assistant", "content":response})
    print(f"AI:{response}")


