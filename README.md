The Language module contains model architecture code for BART, BERT and GPT models using Python and Pytorch.

The code in this repository is meant as reference for those who want to learn how to build transformer based LLMs from scratch. The model architecture is designed for single machine / GPU training, but has the option for multi-machine / GPU distriubuted training if the hardware and resources are available. Otherwise, it is optimized for training smaller scale models with 124M to 170M paramters on a single RT3090 or RTX4090 GPU (24GB of memory).

training.py contains examples for training any of the three model types. chatbot.py with an example UI built with PySide6 to showcase how the GPT models can be fine-tuned as chatbot with three different model examples. It also showcases how a fine-tuned BERT classifier model can also be incorporated into the chatbot UI - displaying sentiment scores and tagging named entities in the chat dialogue.

There are many todo's that I still need to complete for this repository, but in the mean time please feel free to refernece and use the code in your projects as you see fit.

Google Drive link for the saved pre-trained and fine-tuned models referenced in the Language package: https://drive.google.com/drive/folders/1n0Z9NpmaXDtpGO1I3u61v8gUR5RNxZzE?usp=drive_link

I am currently working on building mini-packages for vision and audio models and will commit those once they are complete.

More to follow...enjoy and happy coding!
