import gradio as gr
import pickle

# Loading the model
model = pickle.load(open('sentiment_analysis_model.p','rb'))

# A function which takes a text input and returns the sentiment label
def sentiment_analysis(text):
    sentiment = model.predict([text])
    if sentiment == 1:
        return "Positive"
    else:
        return "Negative"

gradio_ui = gr.Interface(
    fn=sentiment_analysis,
    title="Sentiment Analysis",
    description="Enter some text and see if the model can gauge the sentiment correctly.FLAG if it can't predict correctly.",
    inputs=gr.inputs.Textbox(lines=10, label="Write some text here"),
    outputs=gr.outputs.Textbox(label="Sentiment Label"),
)

gradio_ui.launch(share = 'True')