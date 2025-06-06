from transformers import pipeline
import gradio as gr

# Use a smaller model that works on CPU
pipe = pipeline(
    "text2text-generation", 
    model="google/flan-t5-small", 
    max_length=50
)

def chat_with_llm(prompt):
    response = pipe(prompt)[0]['generated_text']
    return response

iface = gr.Interface(
    fn=chat_with_llm,
    inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
    outputs="text",
    title="LLM Chatbot (Flan-T5 Small)",
    description="Chat with an instruction-tuned model (Flan-T5 Small)."
)

if __name__ == "__main__":
    iface.launch()
