import logging
import telegram
import API
import time
import datetime
import os
import psycopg2

from functools import wraps
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, RegexHandler

FIRST_MODEL_USE, SECOND_MODEL_USE, MODELS_DECLARE = range(3)

class TelegramROBot:

    def show_start_over_keyboard(self, bot, update):
        custom_keyboard = [['Back']]
        reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
        bot.send_message(chat_id=update.message.chat_id, 
                 text="Or you can start over.", 
                 reply_markup=reply_markup)

    def show_start_keyboard(self, bot, update):
        custom_keyboard = [['Random Forests and SVM', 'LSTM'],
                           ['/exit']]
        reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
        bot.send_message(chat_id=update.message.chat_id, 
                 text="Please choose the function.", 
                 reply_markup=reply_markup)

    def first_model_declare(self, bot, update):
        bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
        time.sleep(1)
        bot.send_message(chat_id=update.message.chat_id,
                         text="You chose to use the Random Forests and SVM model.")
        bot.send_message(chat_id=update.message.chat_id,
                         text="Please enter the arguments in format of:")
        bot.send_message(chat_id=update.message.chat_id,
                         text="<higher cost> <lower cost> <second> <minute> <hour> <week day number> <week number> <month number> <DD/MM/YY>")
        self.show_start_over_keyboard(bot, update)
        return FIRST_MODEL_USE

    def second_model_declare(self, bot, update):
        bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
        time.sleep(1)
        bot.send_message(chat_id=update.message.chat_id,
                         text="You chose to use the LSTM model.")
        bot.send_message(chat_id=update.message.chat_id,
                         text="Please enter the arguments in format of:")
        bot.send_message(chat_id=update.message.chat_id,
                         text="<DD/MM/YY> <hour> <minute> <second>")
        self.show_start_over_keyboard(bot, update)
        return SECOND_MODEL_USE

    def models_declare(self, bot, update):
        if update.message.text == 'Random Forests and SVM':
            return self.first_model_declare(bot, update)
        elif update.message.text == 'LSTM':
            return self.second_model_declare(bot, update)
        else:
            bot.send_message(chat_id=update.message.chat_id,
                         text="Sorry, did not understand your model.")
            return MODELS_DECLARE

    def second_model_use(self, bot, update):
        if update.message.text == 'Back':
            self.show_start_keyboard(bot, update)
            return MODELS_DECLARE
        bot.send_message(chat_id=update.message.chat_id,
                         text="Acquiring the information from: LSTM model")
        bot.send_message(chat_id=update.message.chat_id, text=("Predicted price is: %s" % self.facade.get_cost(update.message.text)))
        self.show_start_keyboard(bot, update)
        return MODELS_DECLARE

    def first_model_use(self, bot, update):
        if update.message.text == 'Back':
            self.show_start_keyboard(bot, update)
            return MODELS_DECLARE
        bot.send_message(chat_id=update.message.chat_id,
                         text="Acquiring the information from: Random Forests and SVR model")
        args = list(update.message.text.split(' '))
        args[0] = float(args[0])
        args[1] = float(args[1])
        for i in range(2, 7):
            args[i] = int(args[i])
        args[8] = time.mktime(datetime.datetime.strptime(args[8], "%d/%m/%Y").timetuple())
        result = self.facade.get_probability(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])
        bot.send_message(chat_id=update.message.chat_id,
                         text="The predicted price is: %s" % str(result))
        self.show_start_keyboard(bot, update)
        return MODELS_DECLARE

    def error(self, bot, update, error):
        bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
        time.sleep(1)
        bot.send_message(chat_id=update.message.chat_id, text="Sorry, there is an error occured! :( - %s" % str(error))

    def exit(self, bot, update):
        bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
        time.sleep(1)
        custom_keyboard = [['/start']]
        reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
        bot.send_message(chat_id=update.message.chat_id, 
                 text="Ok, see you later!", 
                 reply_markup=reply_markup)
        return ConversationHandler.END

    def start(self, bot, update):
        self.show_start_keyboard(bot, update)
        return MODELS_DECLARE
        
    def launch(self):
        self.facade = API.Facade()
        self.updater.start_polling()
        self.updater.idle()

    def __init__(self):

        self.updater = Updater(token='742169188:AAExqFAHXxvhYp59d95SJlrg9n_hhZe0vuE')
        self.dispatcher = self.updater.dispatcher

        conv_handler = ConversationHandler(
            entry_points = [CommandHandler('start', self.start)],

            states = {
                MODELS_DECLARE: [RegexHandler('^(Random Forests and SVM|LSTM)$', self.models_declare)],
                FIRST_MODEL_USE: [RegexHandler('^((\d?.\d+) (\d?.\d+) \d{2} \d{2} \d{2} \d \d \d{2} \d{2}/\d{2}/\d{4}|Back)$', self.first_model_use)],
                SECOND_MODEL_USE: [RegexHandler('^(\d{2}/\d{2}/\d{4} \d{2} \d{2} \d{2}|Back)$', self.second_model_use)],
            },

            fallbacks=[CommandHandler('exit', self.exit)]
        )

        self.dispatcher.add_handler(conv_handler)
        self.dispatcher.add_error_handler(self.error)
        
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                             level=logging.INFO)
if __name__ == '__main__':
    bot = TelegramROBot()
    bot.launch()
