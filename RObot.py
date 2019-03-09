from telegram.ext import Updater
import logging
from telegram.ext import CommandHandler
import API

class TelegramROBot:

    def unknown(bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Sorry, I didn't understand that command.")

    def start(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Welcome! Please choose the function.")

    def alive(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Yes, I am alive.")

    def get_probability(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Requesting probability...")
        self.facade.get_

    def get_cost(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Requesting cost...")

    def launch(self):
        self.facade = API.Facade()
        self.updater.start_polling()

    def __init__(self):

        self.REQUEST_KWARGS={
            'proxy_url': 'http://149.28.192.6:8080'
        }

        self.start_handler = CommandHandler('start', self.start)
        self.alive_handler = CommandHandler('alive', self.alive)
        self.unknown_handler = MessageHandler(Filters.command, unknown)

        self.updater = Updater(token='742169188:AAExqFAHXxvhYp59d95SJlrg9n_hhZe0vuE', request_kwargs=self.REQUEST_KWARGS)
        
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(self.start_handler)
        self.dispatcher.add_handler(self.alive_handler)
        self.dispatcher.add_handler(self.unknown_handler)

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                             level=logging.INFO)
if __name__ == '__main__':
    bot = TelegramROBot()
    bot.launch()
