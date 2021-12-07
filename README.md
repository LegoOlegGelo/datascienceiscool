# Telegram bot that solve MNIST problem
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. ([Continue in wikipedia](https://en.wikipedia.org/wiki/MNIST_database))

![MNIST dataset image (Wikipedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/320px-MnistExamples.png)


# Repository structure

## 'model_training' folder
It includes .ipynb file - Jupyter Notebook with model training code.
PyTorch was used for training.

## 'bot' folder
It includes a python scripts:
- main.py - main file with bot logic
- mnist_cnn.py - includes model, linking classes
- mnist_cnn.saved - saved trained model

## bot_settings.json
This file must be next to main.py with json code:
```
{
    "api_token": "<your_api_token>", 
    "log_on": true,
    "save_predictions": true
}
```
parameters:
- api_token - str : telegram bot token (can be found in @BotFather)
- log_on - bool : logging during working (ex. "Prediction for file_55.jpg: 4")
- save_prediction - bool : saving prediction results (if true: need a "predictions_list.csv" file to main.py)


# Model structure:
```
MnistCnn(
  (conv1): Sequential(
    (0): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (lin): Sequential(
    (0): Linear(in_features=400, out_features=120, bias=True)
    (1): ReLU()
    (2): Linear(in_features=120, out_features=84, bias=True)
    (3): ReLU()
  )
  (out): Linear(in_features=84, out_features=10, bias=True)
)
```


# Dependencies
To use the bot:
- pyTelegramBotAPI
- torch
- torchvision
- Pillow


# My bot
You can test my bot @datascienceiscool_bot. But it works periodically (when running on my local machine).
