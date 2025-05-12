import logging
import os
import asyncio
import requests
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

# دریافت توکن تلگرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# دریافت کلیدهای API مجزا برای هر مدل
OPENROUTER_API_KEY_DEEPSEEK = os.getenv("OPENROUTER_API_KEY_DEEPSEEK")
OPENROUTER_API_KEY_GEMMA = os.getenv("OPENROUTER_API_KEY_GEMMA")

# دریافت اطلاعات اختیاری برای هدر OpenRouter
SITE_URL = os.getenv("YOUR_SITE_URL", "http://t.me/your_bot_username") # یک مقدار پیش‌فرض
APP_NAME = os.getenv("YOUR_APP_NAME", "Telegram LLM Bot")      # یک مقدار پیش‌فرض

# --- نام دقیق مدل‌ها در OpenRouter ---
DEEPSEEK_MODEL_NAME = "deepseek/deepseek-chat"
GEMMA_MODEL_NAME = "google/gemma-3-27b-it"

# نگاشت نام مدل به کلید API مربوطه
API_KEYS_MAP = {
    DEEPSEEK_MODEL_NAME: OPENROUTER_API_KEY_DEEPSEEK,
    GEMMA_MODEL_NAME: OPENROUTER_API_KEY_GEMMA,
}

# آدرس API OpenRouter
OPENROUTER_API_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# تنظیمات لاگ‌گیری
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# دیکشنری برای ذخیره مدل انتخاب شده توسط هر کاربر
user_selected_model = {}
# دیکشنری برای ذخیره تاریخچه مکالمات
user_chat_history = {}

async def call_openrouter_api(prompt: str, chat_id: int, model_name: str) -> str:
    """یک درخواست به API OpenRouter با استفاده از کلید API صحیح ارسال می‌کند."""
    api_key = API_KEYS_MAP.get(model_name)

    if not api_key:
        logger.error(f"No API key configured in API_KEYS_MAP for model: {model_name} for chat {chat_id}")
        return f"خطا: کلید API برای مدل '{model_name}' در تنظیمات ربات یافت نشد. لطفاً به ادمین اطلاع دهید."
    if not api_key.startswith("sk-or-"):
         logger.warning(f"API key for model {model_name} does not start with 'sk-or-'. Is it correct?")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME,
    }

    if chat_id not in user_chat_history or user_chat_history[chat_id].get('model') != model_name:
        user_chat_history[chat_id] = {'model': model_name, 'messages': []}
        logger.info(f"Starting/Resetting chat history for user {chat_id} with model {model_name}")

    user_chat_history[chat_id]['messages'].append({"role": "user", "content": prompt})
    max_history_len = 10
    if len(user_chat_history[chat_id]['messages']) > max_history_len:
        user_chat_history[chat_id]['messages'] = user_chat_history[chat_id]['messages'][-max_history_len:]

    payload = {
        "model": model_name,
        "messages": user_chat_history[chat_id]['messages'],
        "max_tokens": 1536,
        "temperature": 0.7,
    }

    logger.debug(f"Sending payload to OpenRouter for chat {chat_id} using model {model_name}: {payload}")

    try:
        response = requests.post(OPENROUTER_API_ENDPOINT, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Received response from OpenRouter for chat {chat_id}: {result}")

        if not result.get("choices") or not result["choices"][0].get("message") or not result["choices"][0]["message"].get("content"):
             logger.error(f"Invalid response structure from OpenRouter for chat {chat_id}: {result}")
             if user_chat_history.get(chat_id) and user_chat_history[chat_id]['messages'] and user_chat_history[chat_id]['messages'][-1]['role'] == 'user':
                  user_chat_history[chat_id]['messages'].pop()
             return "خطا: ساختار پاسخ دریافت شده از OpenRouter نامعتبر است."

        ai_response = result["choices"][0]["message"]["content"].strip()
        user_chat_history[chat_id]['messages'].append({"role": "assistant", "content": ai_response})
        return ai_response

    except requests.exceptions.Timeout:
        logger.error(f"OpenRouter API request timed out for chat {chat_id} with model {model_name}")
        if user_chat_history.get(chat_id) and user_chat_history[chat_id]['messages'] and user_chat_history[chat_id]['messages'][-1]['role'] == 'user':
             user_chat_history[chat_id]['messages'].pop()
        return "خطا: درخواست به OpenRouter زمان زیادی طول کشید و متوقف شد."
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API request error for chat {chat_id} with model {model_name}: {e}")
        error_detail = ""
        if e.response is not None:
            try:
                response_text = e.response.text if e.response.text else "No response body"
                if e.response.status_code == 401:
                    error_detail = " (خطای احراز هویت - کلید API نامعتبر یا منقضی شده است)"
                else:
                    error_detail = f" (کد وضعیت سرور: {e.response.status_code} - {response_text[:200]})"
            except Exception as parse_err:
                logger.error(f"Could not parse error response details: {parse_err}")
                error_detail = f" (کد وضعیت سرور: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'})"
        if user_chat_history.get(chat_id) and user_chat_history[chat_id]['messages'] and user_chat_history[chat_id]['messages'][-1]['role'] == 'user':
             user_chat_history[chat_id]['messages'].pop()
        return f"خطا در ارتباط با OpenRouter: {e}{error_detail}"
    except Exception as e:
        logger.error(f"Error processing OpenRouter response or history for chat {chat_id}: {e}", exc_info=True)
        if user_chat_history.get(chat_id) and user_chat_history[chat_id]['messages'] and user_chat_history[chat_id]['messages'][-1]['role'] == 'user':
             user_chat_history[chat_id]['messages'].pop()
        return f"یک خطای غیرمنتظره در پردازش رخ داد: {e}"

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """نمایش منوی اصلی با گزینه‌های مختلف"""
    keyboard = [
        [InlineKeyboardButton("🔘 انتخاب مدل هوش مصنوعی", callback_data="change_model")],
        [InlineKeyboardButton("ℹ️ اطلاعات ربات", callback_data="bot_info")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    chat_id = update.effective_chat.id
    current_model = user_selected_model.get(chat_id, "هیچ مدلی انتخاب نشده")
    
    message = f"🏠 منوی اصلی:\n\n🔹 مدل فعلی: {current_model}\n\nلطفاً یک گزینه را انتخاب کنید:"
    
    if update.callback_query:
        await update.callback_query.edit_message_text(text=message, reply_markup=reply_markup)
    else:
        await update.message.reply_text(text=message, reply_markup=reply_markup)

async def handle_menu_actions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """مدیریت کلیک روی گزینه‌های منو"""
    query = update.callback_query
    await query.answer()
    
    chat_id = query.message.chat_id
    action = query.data
    
    if action == "change_model":
        await show_model_selection(update, context)
    elif action == "bot_info":
        # نمایش اطلاعات ربات با مدت زمان بیشتر
        info_text = (
            "🤖 اطلاعات ربات:\n\n"
            "این ربات از طریق سرویس OpenRouter به مدل‌های هوش مصنوعی متصل می‌شود.\n\n"
            "🔹 مدل‌های پشتیبانی شده:\n"
            f"- {DEEPSEEK_MODEL_NAME}\n"
            f"- {GEMMA_MODEL_NAME}\n\n"
            "برای تغییر مدل از منوی اصلی گزینه 'تغییر مدل' را انتخاب کنید."
        )
        await query.edit_message_text(text=info_text)
        await asyncio.sleep(2)  # افزایش زمان نمایش به 2 ثانیه
        await show_main_menu(update, context)
    elif action == "back_to_menu":
        await show_main_menu(update, context)

async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """نمایش منوی انتخاب مدل"""
    query = update.callback_query
    chat_id = query.message.chat_id
    
    keyboard = []
    if API_KEYS_MAP.get(DEEPSEEK_MODEL_NAME):
        keyboard.append([InlineKeyboardButton("🤖 DeepSeek", callback_data=DEEPSEEK_MODEL_NAME)])
    
    if API_KEYS_MAP.get(GEMMA_MODEL_NAME):
        keyboard.append([InlineKeyboardButton("✨ Gemma", callback_data=GEMMA_MODEL_NAME)])
    
    keyboard.append([InlineKeyboardButton("🔙 بازگشت به منوی اصلی", callback_data="back_to_menu")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text="لطفاً یکی از مدل‌های زبانی زیر را انتخاب کنید:",
        reply_markup=reply_markup
    )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """دستور /start را مدیریت می‌کند و منوی اصلی را نمایش می‌دهد."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    logger.info(f"User {user.id} ({user.first_name}) started the bot in chat {chat_id}.")

    # پاک کردن تاریخچه و انتخاب قبلی کاربر با دستور استارت
    if chat_id in user_chat_history:
        user_chat_history[chat_id]['messages'] = []
    if chat_id in user_selected_model:
        del user_selected_model[chat_id]

    await show_main_menu(update, context)

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """دستور /menu را مدیریت می‌کند و منوی اصلی را نمایش می‌دهد."""
    await show_main_menu(update, context)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """پاسخ به کلیک روی دکمه‌های InlineKeyboard."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    selected_model_name = query.data

    # اگر داده callback مربوط به منو باشد
    if selected_model_name in ["change_model", "bot_info", "back_to_menu"]:
        await handle_menu_actions(update, context)
        return

    # بررسی مجدد وجود کلید API برای مدل انتخاب شده
    if not API_KEYS_MAP.get(selected_model_name):
        logger.error(f"User {chat_id} selected model {selected_model_name}, but API key is missing!")
        await query.edit_message_text(
            text=f"خطا: کلید API برای مدل '{selected_model_name}' در دسترس نیست. لطفاً مدل دیگری را انتخاب کنید یا به ادمین اطلاع دهید."
        )
        if chat_id in user_selected_model:
            del user_selected_model[chat_id]
        return

    user_selected_model[chat_id] = selected_model_name
    if chat_id not in user_chat_history:
        user_chat_history[chat_id] = {'model': selected_model_name, 'messages': []}
    else:
        user_chat_history[chat_id]['model'] = selected_model_name
        user_chat_history[chat_id]['messages'] = []
    
    logger.info(f"User {chat_id} selected model: {selected_model_name}")

    friendly_model_name = "مدل انتخاب شده"
    if selected_model_name == DEEPSEEK_MODEL_NAME:
        friendly_model_name = "DeepSeek"
    elif selected_model_name == GEMMA_MODEL_NAME:
        friendly_model_name = "Gemma"

    await query.edit_message_text(
        text=f"شما مدل {friendly_model_name} ({selected_model_name}) را انتخاب کردید.\nحالا می‌توانید پیام خود را ارسال کنید."
    )
    await show_main_menu(update, context)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """پیام‌های متنی کاربر را پردازش می‌کند و به OpenRouter ارسال می‌کند."""
    chat_id = update.message.chat_id
    user_text = update.message.text

    logger.info(f"Message from {chat_id}: '{user_text}'")

    if chat_id not in user_selected_model:
        await update.message.reply_text("لطفاً ابتدا یک مدل را از منوی ربات انتخاب کنید. (/menu)")
        return

    selected_model = user_selected_model[chat_id]

    if not API_KEYS_MAP.get(selected_model):
        logger.error(f"Attempted to send message from {chat_id} using model {selected_model}, but API key is missing!")
        await update.message.reply_text(f"خطا: کلید API برای مدل '{selected_model}' در دسترس نیست. لطفاً با /menu مدل دیگری را انتخاب کنید یا به ادمین اطلاع دهید.")
        del user_selected_model[chat_id]
        if chat_id in user_chat_history:
             del user_chat_history[chat_id]
        return

    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    response_text = await call_openrouter_api(user_text, chat_id, selected_model)

    try:
        await update.message.reply_text(response_text)
    except Exception as e:
        logger.error(f"Failed to send message to chat {chat_id}: {e}")
        await update.message.reply_text("متاسفانه مشکلی در ارسال پاسخ به وجود آمد.")

def main() -> None:
    """ربات را راه‌اندازی و اجرا می‌کند."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("توکن ربات تلگرام یافت نشد. متغیر TELEGRAM_BOT_TOKEN را در فایل .env تنظیم کنید.")
        return

    # بررسی وجود کلیدهای API در هنگام شروع
    keys_found = False
    if not OPENROUTER_API_KEY_DEEPSEEK:
        logger.warning(f"کلید API OpenRouter برای مدل DeepSeek ({DEEPSEEK_MODEL_NAME}) یافت نشد. این مدل در دسترس نخواهد بود.")
    else:
        if not OPENROUTER_API_KEY_DEEPSEEK.startswith("sk-or-"):
            logger.warning(f"کلید API DeepSeek ({DEEPSEEK_MODEL_NAME}) با 'sk-or-' شروع نمی‌شود.")
        keys_found = True

    if not OPENROUTER_API_KEY_GEMMA:
        logger.warning(f"کلید API OpenRouter برای مدل Gemma ({GEMMA_MODEL_NAME}) یافت نشد. این مدل در دسترس نخواهد بود.")
    else:
         if not OPENROUTER_API_KEY_GEMMA.startswith("sk-or-"):
            logger.warning(f"کلید API Gemma ({GEMMA_MODEL_NAME}) با 'sk-or-' شروع نمی‌شود.")
         keys_found = True

    if not keys_found:
        logger.critical("هیچ کلید API معتبری برای OpenRouter یافت نشد. ربات نمی‌تواند به مدل‌ها متصل شود و اجرا نخواهد شد.")
        return

    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    application.run_polling()
    logger.info("Bot stopped.")

if __name__ == "__main__":
    main()
