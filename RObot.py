import logging
from telegram.ext import CommandHandler, Filters, Updater, MessageHandler
import telegram
import API
import time
import datetime

class TelegramROBot:

    def request(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id, text="Interpretating...")

    def start(self, bot, update):
        #input_message = update.message.text
        
        custom_keyboard = [['first_module', 'second_module']]
        reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
        bot.send_message(chat_id=update.message.chat_id, 
                 text="Welcome! Please choose the function.", 
                 reply_markup=reply_markup)

    #def get_cost_from_first(self, bot, update):
     #   pass
        #args = list(map(float, args))
        #result = self.facade.get_probability(args[0], args[1], args[2],
        #                                args[3], args[4], args[5], args[6], args[7],
        #                                time.mktime(datetime.datetime.strptime(args[8], "%d/%m/%Y").timetuple()))
        #bot.send_message(chat_id=update.message.chat_id, text="The cost of first model is: {0} ETH" % str(result))

    def launch(self):
        self.facade = API.Facade()
        self.updater.start_polling()

    def __init__(self):

        self.REQUEST_KWARGS={
            'proxy_url': 'socks5://110.49.101.58:1080'
        }

        self.start_handler = CommandHandler('start', self.start)        
        self.request_handler = MessageHandler(Filters.command, self.request)

        self.updater = Updater(token='742169188:AAExqFAHXxvhYp59d95SJlrg9n_hhZe0vuE', request_kwargs=self.REQUEST_KWARGS)
        
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(self.start_handler)
        self.dispatcher.add_handler(self.request_handler)
        
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                             level=logging.INFO)
if __name__ == '__main__':
    bot = TelegramROBot()
    bot.launch()
