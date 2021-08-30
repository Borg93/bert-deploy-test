import torch
from transformers import pipeline



path = "roberta-base"

class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, text):
        unmasker = pipeline('fill-mask', model=path)
        return unmasker(text)


model = Model()

def get_model():
    return model
