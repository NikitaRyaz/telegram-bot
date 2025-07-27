import os
import logging
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext, MessageHandler, Filters
from binance.client import Client
from binance.enums import *
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Инициализация клиента Binance
client = Client(API_KEY, API_SECRET)

class TradingBot:
    def __init__(self):
        self.risk_level = 0.02  # 2% риска на сделку по умолчанию
        self.take_profit = 0.03  # 3% тейк-профит
        self.stop_loss = 0.015  # 1.5% стоп-лосс
        self.trading_enabled = False
        self.asset = 'BTC'
        self.base_currency = 'USDT'
        self.symbol = f"{self.asset}{self.base_currency}"
        self.timeframe = '1h'
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20

    def get_account_balance(self, asset: str) -> float:
        """Получить баланс аккаунта по конкретному активу"""
        try:
            balance = client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except Exception as e:
            logger.error(f"Error getting balance for {asset}: {e}")
            return 0.0

    def get_current_price(self, symbol: str) -> float:
        """Получить текущую цену пары"""
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_historical_data(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Получить исторические данные для анализа"""
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.apply(pd.to_numeric, errors='ignore')
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать технические индикаторы"""
        # RSI
        df['rsi'] = RSIIndicator(df['close'], window=self.rsi_period).rsi()
        
        # MACD
        macd = MACD(df['close'], window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=self.bb_period, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df

    def analyze_market(self) -> dict:
        """Анализ рынка и генерация торговых сигналов"""
        df = self.get_historical_data(self.symbol, self.timeframe)
        if df.empty:
            return {'signal': 'neutral', 'confidence': 0}
        
        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]
        
        signals = []
        confidence = 0
        
        # RSI анализ
        if last_row['rsi'] > 70:
            signals.append('overbought')
            confidence += 0.3
        elif last_row['rsi'] < 30:
            signals.append('oversold')
            confidence += 0.3
        
        # MACD анализ
        if last_row['macd'] > last_row['macd_signal'] and df.iloc[-2]['macd'] < df.iloc[-2]['macd_signal']:
            signals.append('macd_bullish')
            confidence += 0.2
        elif last_row['macd'] < last_row['macd_signal'] and df.iloc[-2]['macd'] > df.iloc[-2]['macd_signal']:
            signals.append('macd_bearish')
            confidence += 0.2
        
        # Bollinger Bands анализ
        if last_row['close'] < last_row['bb_lower']:
            signals.append('bb_oversold')
            confidence += 0.2
        elif last_row['close'] > last_row['bb_upper']:
            signals.append('bb_overbought')
            confidence += 0.2
        
        # Определение общего сигнала
        if not signals:
            return {'signal': 'neutral', 'confidence': 0}
        
        bullish = signals.count('oversold') + signals.count('macd_bullish') + signals.count('bb_oversold')
        bearish = signals.count('overbought') + signals.count('macd_bearish') + signals.count('bb_overbought')
        
        if bullish > bearish:
            return {'signal': 'buy', 'confidence': min(confidence, 1.0)}
        elif bearish > bullish:
            return {'signal': 'sell', 'confidence': min(confidence, 1.0)}
        else:
            return {'signal': 'neutral', 'confidence': 0}

    def calculate_position_size(self, current_price: float) -> float:
        """Рассчитать размер позиции на основе уровня риска"""
        account_balance = self.get_account_balance(self.base_currency)
        risk_amount = account_balance * self.risk_level
        stop_loss_amount = current_price * self.stop_loss
        position_size = risk_amount / stop_loss_amount
        return min(position_size, account_balance / current_price)

    def execute_sell_order(self, current_price: float, reason: str) -> bool:
        """Выполнить ордер на продажу"""
        try:
            balance = self.get_account_balance(self.asset)
            if balance <= 0:
                logger.warning(f"No {self.asset} to sell")
                return False
            
            # Рассчитать размер позиции с учетом риска
            quantity = round(self.calculate_position_size(current_price), 6)
            if quantity <= 0:
                logger.warning("Invalid quantity for sell order")
                return False
            
            # Проверить минимальный размер ордера
            info = client.get_symbol_info(self.symbol)
            min_qty = float(next(filter(lambda f: f['filterType'] == 'LOT_SIZE', info['filters']))['minQty'])
            if quantity < min_qty:
                logger.warning(f"Quantity {quantity} is less than minimum {min_qty}")
                return False
            
            # Разместить лимитный ордер с тейк-профитом
            take_profit_price = round(current_price * (1 + self.take_profit), 2)
            
            # Основной ордер на продажу
            order = client.create_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=current_price
            )
            
            logger.info(f"Sold {quantity} {self.asset} at {current_price} {self.base_currency}. Reason: {reason}")
            
            # Разместить тейк-профит ордер (OCO - One Cancels Other)
            try:
                client.create_oco_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    quantity=quantity,
                    price=take_profit_price,
                    stopPrice=current_price * (1 - self.stop_loss),
                    stopLimitPrice=current_price * (1 - self.stop_loss),
                    stopLimitTimeInForce=TIME_IN_FORCE_GTC
                )
                logger.info(f"Take profit set at {take_profit_price} and stop loss at {current_price * (1 - self.stop_loss)}")
            except Exception as e:
                logger.error(f"Error setting OCO order: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            return False

    def check_and_execute_trades(self):
        """Проверить условия и выполнить торговые операции"""
        if not self.trading_enabled:
            return
        
        analysis = self.analyze_market()
        current_price = self.get_current_price(self.symbol)
        
        if analysis['signal'] == 'sell' and analysis['confidence'] > 0.5:
            self.execute_sell_order(current_price, f"Strong sell signal (confidence: {analysis['confidence']:.2f})")
        elif analysis['signal'] == 'sell' and analysis['confidence'] > 0.3:
            # Для менее уверенных сигналов продаем только часть
            original_risk = self.risk_level
            self.risk_level *= 0.5  # Уменьшаем риск вдвое
            self.execute_sell_order(current_price, f"Moderate sell signal (confidence: {analysis['confidence']:.2f})")
            self.risk_level = original_risk

# Инициализация торгового бота
bot = TradingBot()

# Обработчики команд Telegram
def start(update: Update, context: CallbackContext) -> None:
    """Отправить приветственное сообщение при команде /start"""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr"Привет {user.mention_markdown_v2()}\! Я бот для автоматической продажи криптовалюты\. "
        "Используйте команды для управления:\n\n"
        "/status \- Текущий статус и настройки\n"
        "/enable \- Включить автоматическую торговлю\n"
        "/disable \- Выключить автоматическую торговлю\n"
        "/settings \- Настройки параметров торговли\n"
        "/balance \- Показать баланс\n"
        "/analyze \- Анализ текущего рынка"
    )

def status(update: Update, context: CallbackContext) -> None:
    """Показать текущий статус бота"""
    status_text = (
        f"🔄 Статус бота:\n\n"
        f"🔹 Автоторговля: {'✅ Включена' if bot.trading_enabled else '❌ Выключена'}\n"
        f"🔹 Торговая пара: {bot.symbol}\n"
        f"🔹 Таймфрейм: {bot.timeframe}\n"
        f"🔹 Уровень риска: {bot.risk_level*100:.1f}%\n"
        f"🔹 Тейк-профит: {bot.take_profit*100:.1f}%\n"
        f"🔹 Стоп-лосс: {bot.stop_loss*100:.1f}%\n\n"
        f"📊 Технические индикаторы:\n"
        f"  • RSI период: {bot.rsi_period}\n"
        f"  • MACD: {bot.macd_fast}/{bot.macd_slow}/{bot.macd_signal}\n"
        f"  • Bollinger Bands: {bot.bb_period} период\n"
    )
    
    # Добавить текущий анализ рынка
    analysis = bot.analyze_market()
    current_price = bot.get_current_price(bot.symbol)
    
    status_text += (
        f"\n📈 Текущий анализ:\n"
        f"🔹 Цена: {current_price:.2f} {bot.base_currency}\n"
        f"🔹 Сигнал: {analysis['signal']}\n"
        f"🔹 Уверенность: {analysis['confidence']*100:.1f}%\n"
    )
    
    update.message.reply_text(status_text)

def enable_trading(update: Update, context: CallbackContext) -> None:
    """Включить автоматическую торговлю"""
    bot.trading_enabled = True
    update.message.reply_text("✅ Автоматическая торговля включена. Бот будет мониторить рынок и выполнять сделки согласно стратегии.")

def disable_trading(update: Update, context: CallbackContext) -> None:
    """Выключить автоматическую торговлю"""
    bot.trading_enabled = False
    update.message.reply_text("❌ Автоматическая торговля выключена. Бот не будет выполнять никаких сделок.")

def show_balance(update: Update, context: CallbackContext) -> None:
    """Показать баланс аккаунта"""
    asset_balance = bot.get_account_balance(bot.asset)
    base_balance = bot.get_account_balance(bot.base_currency)
    current_price = bot.get_current_price(bot.symbol)
    total_value = asset_balance * current_price + base_balance
    
    balance_text = (
        f"💰 Баланс аккаунта:\n\n"
        f"🔹 {bot.asset}: {asset_balance:.6f}\n"
        f"🔹 {bot.base_currency}: {base_balance:.2f}\n"
        f"🔹 Текущая цена: {current_price:.2f} {bot.base_currency}\n"
        f"🔹 Общая стоимость: {total_value:.2f} {bot.base_currency}"
    )
    update.message.reply_text(balance_text)

def analyze_market(update: Update, context: CallbackContext) -> None:
    """Показать текущий анализ рынка"""
    analysis = bot.analyze_market()
    current_price = bot.get_current_price(bot.symbol)
    df = bot.get_historical_data(bot.symbol, bot.timeframe)
    df = bot.calculate_indicators(df)
    last_row = df.iloc[-1]
    
    analysis_text = (
        f"📊 Анализ рынка для {bot.symbol} ({bot.timeframe}):\n\n"
        f"🔹 Текущая цена: {current_price:.2f}\n"
        f"🔹 Сигнал: {analysis['signal'].upper()} (уверенность: {analysis['confidence']*100:.1f}%)\n\n"
        f"📈 Технические индикаторы:\n"
        f"  • RSI: {last_row['rsi']:.2f}\n"
        f"  • MACD: {last_row['macd']:.4f}\n"
        f"  • Signal: {last_row['macd_signal']:.4f}\n"
        f"  • Hist: {last_row['macd_diff']:.4f}\n"
        f"  • BB Width: {last_row['bb_width']:.4f}\n"
        f"  • Price vs BB: {last_row['close']:.2f} (L:{last_row['bb_lower']:.2f}, M:{last_row['bb_middle']:.2f}, U:{last_row['bb_upper']:.2f})\n\n"
    )
    
    # Добавить интерпретацию
    interpretation = ""
    if analysis['signal'] == 'sell':
        interpretation = "📉 Сигнал на продажу. Рынок может быть перекуплен или начинается нисходящий тренд."
    elif analysis['signal'] == 'buy':
        interpretation = "📈 Сигнал на покупку. Рынок может быть перепродан или начинается восходящий тренд."
    else:
        interpretation = "➖ Нейтральный сигнал. Рынок в консолидации или сигналы противоречивы."
    
    update.message.reply_text(analysis_text + interpretation)

def settings_menu(update: Update, context: CallbackContext) -> None:
    """Показать меню настроек"""
    keyboard = [
        [InlineKeyboardButton("Изменить уровень риска", callback_data='set_risk')],
        [InlineKeyboardButton("Изменить тейк-профит", callback_data='set_tp')],
        [InlineKeyboardButton("Изменить стоп-лосс", callback_data='set_sl')],
        [InlineKeyboardButton("Изменить торговую пару", callback_data='set_pair')],
        [InlineKeyboardButton("Изменить таймфрейм", callback_data='set_tf')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('⚙️ Настройки торгового бота:', reply_markup=reply_markup)

def button_handler(update: Update, context: CallbackContext) -> None:
    """Обработчик нажатий кнопок"""
    query = update.callback_query
    query.answer()
    
    if query.data == 'set_risk':
        query.edit_message_text(text=f"Текущий уровень риска: {bot.risk_level*100:.1f}%\nОтправьте новое значение (например, 2 для 2%):")
        context.user_data['setting'] = 'risk'
    elif query.data == 'set_tp':
        query.edit_message_text(text=f"Текущий тейк-профит: {bot.take_profit*100:.1f}%\nОтправьте новое значение (например, 3 для 3%):")
        context.user_data['setting'] = 'tp'
    elif query.data == 'set_sl':
        query.edit_message_text(text=f"Текущий стоп-лосс: {bot.stop_loss*100:.1f}%\nОтправьте новое значение (например, 1.5 для 1.5%):")
        context.user_data['setting'] = 'sl'
    elif query.data == 'set_pair':
        query.edit_message_text(text=f"Текущая торговая пара: {bot.symbol}\nОтправьте новую пару (например, ETHUSDT):")
        context.user_data['setting'] = 'pair'
    elif query.data == 'set_tf':
        query.edit_message_text(text=f"Текущий таймфрейм: {bot.timeframe}\nОтправьте новый таймфрейм (например, 1h, 4h, 1d):")
        context.user_data['setting'] = 'tf'

def handle_settings_input(update: Update, context: CallbackContext) -> None:
    """Обработчик ввода настроек"""
    if 'setting' not in context.user_data:
        return
    
    setting = context.user_data['setting']
    text = update.message.text
    
    try:
        if setting == 'risk':
            value = float(text) / 100
            if 0 < value <= 0.1:  # Максимальный риск 10%
                bot.risk_level = value
                update.message.reply_text(f"✅ Уровень риска установлен на {value*100:.1f}%")
            else:
                update.message.reply_text("❌ Уровень риска должен быть между 0.1% и 10%")
        
        elif setting == 'tp':
            value = float(text) / 100
            if value > 0:
                bot.take_profit = value
                update.message.reply_text(f"✅ Тейк-профит установлен на {value*100:.1f}%")
            else:
                update.message.reply_text("❌ Тейк-профит должен быть положительным")
        
        elif setting == 'sl':
            value = float(text) / 100
            if 0 < value < 0.1:  # Максимальный стоп-лосс 10%
                bot.stop_loss = value
                update.message.reply_text(f"✅ Стоп-лосс установлен на {value*100:.1f}%")
            else:
                update.message.reply_text("❌ Стоп-лосс должен быть между 0.1% и 10%")
        
        elif setting == 'pair':
            symbol = text.upper()
            try:
                # Проверить, что пара существует
                client.get_symbol_ticker(symbol=symbol)
                bot.symbol = symbol
                bot.asset = symbol.replace('USDT', '')
                update.message.reply_text(f"✅ Торговая пара установлена на {symbol}")
            except:
                update.message.reply_text("❌ Неверная торговая пара. Убедитесь, что она существует на Binance.")
        
        elif setting == 'tf':
            valid_tf = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            if text in valid_tf:
                bot.timeframe = text
                update.message.reply_text(f"✅ Таймфрейм установлен на {text}")
            else:
                update.message.reply_text(f"❌ Неверный таймфрейм. Допустимые значения: {', '.join(valid_tf)}")
    
    except ValueError:
        update.message.reply_text("❌ Пожалуйста, введите числовое значение")
    
    context.user_data.pop('setting', None)

def error_handler(update: Update, context: CallbackContext) -> None:
    """Обработчик ошибок"""
    logger.error(msg="Exception while handling update:", exc_info=context.error)
    
    if update and update.message:
        update.message.reply_text("⚠️ Произошла ошибка. Пожалуйста, попробуйте позже или проверьте логи.")

def main() -> None:
    """Запуск бота"""
    # Создать Updater и передать токен бота
    updater = Updater(TELEGRAM_TOKEN)
    
    # Получить диспетчер для регистрации обработчиков
    dispatcher = updater.dispatcher
    
    # Регистрация обработчиков команд
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("status", status))
    dispatcher.add_handler(CommandHandler("enable", enable_trading))
    dispatcher.add_handler(CommandHandler("disable", disable_trading))
    dispatcher.add_handler(CommandHandler("balance", show_balance))
    dispatcher.add_handler(CommandHandler("analyze", analyze_market))
    dispatcher.add_handler(CommandHandler("settings", settings_menu))
    
    # Обработчики кнопок и сообщений
    dispatcher.add_handler(CallbackQueryHandler(button_handler))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_settings_input))
    
    # Обработчик ошибок
    dispatcher.add_error_handler(error_handler)
    
    # Запустить бота
    updater.start_polling()
    
    # Запустить проверку рынка по расписанию
    job_queue = updater.job_queue
    job_queue.run_repeating(
        callback=lambda ctx: bot.check_and_execute_trades(),
        interval=300,  # Проверять каждые 5 минут
        first=10
    )
    
    # Запустить бота до прерывания
    updater.idle()

if __name__ == '__main__':
    main()