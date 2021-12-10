import telebot
import json

from mnist_cnn import PredictiveModel


# load bot settings (api_token, log_on mode, ...)
json_file = open('bot_settings.json', 'r')
bot_settings = json.loads(json_file.read())
json_file.close()

TOKEN = bot_settings['api_token']
LOG_ON = bot_settings['log_on']
SAVE_PREDICTIONS = bot_settings['save_predictions']

bot = telebot.TeleBot(TOKEN)


# load CNN model
cnn = PredictiveModel('mnist_cnn.saved')
predictions_list_file = 'predictions_list.csv'


# bot logic
start_msg = "Привет ✌️\n\nПришли мне квадратную (хотя бы приблизительно) фотографию с белым фоном и черной цифрой!"


@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.send_message(message.chat.id, start_msg)


@bot.message_handler(content_types=['photo'])
def image(message):
    # loading and saving image
    fileID = message.photo[0].file_id
    f_path = bot.get_file(fileID).file_path
    downloaded_file = bot.download_file(f_path)

    new_filename = f'bot_images/{f_path}.jpg'

    with open(new_filename, 'wb') as new_file:
        new_file.write(downloaded_file)

    # prediction and output
    prediction = cnn.predict(new_filename)
    
    if LOG_ON:
        print(f'Prediction for {new_filename}: {prediction}.')
    
    bot.send_message(message.chat.id, f'На картинке изображена цифра {prediction}')

    # saving predictions
    if SAVE_PREDICTIONS:
        with open(predictions_list_file, 'a+') as f:
            f.write(f'\n{new_filename},{prediction}')


bot.infinity_polling()
