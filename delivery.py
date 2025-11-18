import os
from twilio.rest import Client
from flask_mail import Mail, Message
import logging

# Twilio credentials (set these in environment variables)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Email configuration
MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
MAIL_USERNAME = os.getenv('MAIL_USERNAME')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

# Initialize Flask-Mail (will be configured in app.py)
mail = Mail()

def send_whatsapp_message(to_number, message, pdf_path=None):
    """Send WhatsApp message with optional PDF attachment."""
    if not twilio_client:
        logging.error("Twilio not configured")
        return False

    try:
        # WhatsApp numbers should be in format: whatsapp:+1234567890
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'

        message_data = {
            'from_': f'whatsapp:{TWILIO_PHONE_NUMBER}',
            'body': message,
            'to': to_number
        }

        if pdf_path and os.path.exists(pdf_path):
            message_data['media_url'] = [f'file://{pdf_path}']

        twilio_client.messages.create(**message_data)
        logging.info(f"WhatsApp message sent to {to_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to send WhatsApp message: {e}")
        return False

def send_sms(to_number, message):
    """Send SMS message."""
    if not twilio_client:
        logging.error("Twilio not configured")
        return False

    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        logging.info(f"SMS sent to {to_number}")
        return True
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")
        return False

def send_email(to_email, subject, message, pdf_path=None):
    """Send email with optional PDF attachment."""
    try:
        msg = Message(subject, sender=MAIL_USERNAME, recipients=[to_email])
        msg.body = message

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                msg.attach(os.path.basename(pdf_path), 'application/pdf', f.read())

        mail.send(msg)
        logging.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False

def deliver_report(prediction_data, pdf_path, contact_info):
    """Deliver report via WhatsApp and SMS only."""
    plant = prediction_data['plant']
    disease = prediction_data['disease']
    confidence = prediction_data['confidence']

    message = f"Plant Disease Prediction Report\n\nPlant: {plant}\nDisease: {disease}\nConfidence: {confidence:.2f}%\n\nPlease check the attached PDF for detailed information."

    success_channels = []

    # WhatsApp
    if 'whatsapp' in contact_info and contact_info['whatsapp']:
        if send_whatsapp_message(contact_info['whatsapp'], message, pdf_path):
            success_channels.append('WhatsApp')

    # SMS
    if 'sms' in contact_info and contact_info['sms']:
        if send_sms(contact_info['sms'], message):
            success_channels.append('SMS')

    return success_channels
