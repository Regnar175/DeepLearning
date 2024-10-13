This repository is meant as a resource for those who want a relatively easy to understand reference for the more common transformer-based AI models currently being used. It attempts to simplify some of the more complex model architectures currently available for reference on GitHub such as HuggingFace or NVIDIA repositories.

The Language mini-package contains model architecture code for BART, BERT and GPT models using Python and Pytorch. The tokenizers and dataset builder functions use code provided by HuggingFace's tokenizers and dataset packages. Otherwise, the models and utility code in this repository are from scratch (as in built using PyTorch and not another intermediate package). It borrows code snippets and ideas from various sources such as HF, NVIDIA, Fairseq, and Karpathy's GitHub repositories. The code is meant as reference for those who want to learn how to build transformer-based LLMs from scratch. The model architectures are designed and optimized for training smaller scale models with 124M to 170M parameters on a single RTX3090 or RTX4090 GPU (24GB of memory), but has the option to scale up for multi-machine / GPU distributed training if the hardware and resources are available.

The training.py module contains examples for training any of the three model types with 4 lines of code. Please make sure to reference the loader and trainer classes in each of the model type sub-directories (base loader and trainer classes are in the utilities.py module).

![code](https://github.com/user-attachments/assets/4570b762-05ee-4919-a4c8-db5d2e5f4071)

The chatbot.py module is an UI built with PySide6 to showcase the fine-tuned GPT models in a user prompt and chatbot response configuration. It also showcases how a fine-tuned BERT classifier model can be incorporated with an uncased GPT model - formatting the output text, displaying sentiment scores on the UI, and tagging named entities in the chat dialogue.

![chatbot](https://github.com/user-attachments/assets/f4ab2a0a-b427-4c5b-a72d-9c9054c23044)

For obvious reasons, the datasets used to pre-train and fine-tune the models are not provided except for two small fine-tuning datasets (BART Grammatical Error Correction (GEC) dataset and a BERT combined classifier dataset for Named Entity Recognition (NER) and Sentiment). Follow the basic instructions / comments provided in the data_prep.py and data_builder.py modules on preparing and building datasets using HuggingFace hub and dataset packages - simply copy the function of the dataset you want to build and run the code in a standalone .py file. The data loader classes are set up for streaming in file chunks instead of loading large files into memory. This is the reason for the file and data indexing attributes which allows for segmented training (e.g. 100,000 iterations in 6-hour chunks vs all 1M iterations continuously without break).

Google Drive link for the saved pre-trained and fine-tuned models referenced throughout the Language mini-package: https://drive.google.com/drive/folders/1n0Z9NpmaXDtpGO1I3u61v8gUR5RNxZzE?usp=drive_link

There are many todo's that I still need to complete for this repository, but in the mean-time please feel free to reference and use the code in your projects as you see fit.

I am currently working on building mini-packages for vision and audio models and will commit those once they are complete.

More to follow...enjoy and happy coding!
