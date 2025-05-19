# daily_tools/communication.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Email Functionality ---
def send_email_actual(to_email: str, subject: str, body: str) -> str:
    """Sends an actual email using SMTP. Requires environment variables for configuration."""
    
    smtp_host = os.getenv("EMAIL_HOST")
    smtp_port_str = os.getenv("EMAIL_PORT")
    smtp_user = os.getenv("EMAIL_HOST_USER")
    smtp_password = os.getenv("EMAIL_HOST_PASSWORD")
    use_tls_str = os.getenv("EMAIL_USE_TLS", "true").lower()
    from_email = smtp_user # Usually, the sender is the authenticated user

    if not all([smtp_host, smtp_port_str, smtp_user, smtp_password, from_email]):
        missing_configs = []
        if not smtp_host: missing_configs.append("EMAIL_HOST")
        if not smtp_port_str: missing_configs.append("EMAIL_PORT")
        if not smtp_user: missing_configs.append("EMAIL_HOST_USER (for sender and login)")
        if not smtp_password: missing_configs.append("EMAIL_HOST_PASSWORD")
        return f"Email not sent. Missing configuration(s): {', '.join(missing_configs)}. Please set environment variables."

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        return "Email not sent. Invalid EMAIL_PORT. It must be an integer."

    use_tls = use_tls_str == "true"

    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = None
        if use_tls:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.ehlo() # Extended Hello
            server.starttls() # Start TLS encryption
            server.ehlo() # Re-identify ourselves as an ESMTP client after starting TLS
        else: # Potentially for SSL on a different port, or no encryption (not recommended)
            # SSL usually uses smtplib.SMTP_SSL() directly with port 465
            if smtp_port == 465: # Common SSL port
                 server = smtplib.SMTP_SSL(smtp_host, smtp_port)
            else: # Assuming non-encrypted or user has specific setup for non-TLS/non-SSL
                 server = smtplib.SMTP(smtp_host, smtp_port)

        server.login(smtp_user, smtp_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return f"Email successfully sent to {to_email} with subject '{subject}'."

    except smtplib.SMTPAuthenticationError as e:
        error_message = f"SMTP Authentication Error: {e.smtp_code} - {e.smtp_error.decode() if e.smtp_error else 'Unknown auth error'}. Check credentials (EMAIL_HOST_USER, EMAIL_HOST_PASSWORD) and SMTP settings."
        print(error_message)
        return error_message
    except smtplib.SMTPServerDisconnected as e:
        error_message = f"SMTP Server Disconnected: {e}. Check EMAIL_HOST, EMAIL_PORT, and network."
        print(error_message)
        return error_message
    except smtplib.SMTPConnectError as e:
        error_message = f"SMTP Connection Error: {e}. Check EMAIL_HOST and EMAIL_PORT."
        print(error_message)
        return error_message
    except smtplib.SMTPException as e:
        error_message = f"SMTP Error: {e}. Please check your email configuration and network."
        print(error_message)
        return error_message
    except ConnectionRefusedError as e:
        error_message = f"Connection Refused: {e}. Ensure SMTP server is running and accessible at {smtp_host}:{smtp_port}."
        print(error_message)
        return error_message
    except TimeoutError as e:
        error_message = f"Connection Timeout: {e}. Server at {smtp_host}:{smtp_port} did not respond in time."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Failed to send email due to an unexpected error: {type(e).__name__} - {e}"
        print(error_message)
        return error_message

# --- Mock Email and Message Functions (Kept for reference or fallback) ---
def send_email_mock(to: str, subject: str, body: str) -> str:
    """Mocks sending an email."""
    # print(f"MOCK: Email to {to}, subject '{subject}', body '{body}'") # Original print
    return f"Mock: Email to {to} with subject '{subject}' would have been sent."

def send_message_mock(to_contact: str, message_body: str) -> str:
    """Mocks sending a message. Actual implementation requires a specific service (e.g., Twilio)."""
    # print(f"MOCK: Message to {to_contact}: '{message_body}'") # Original print
    return f"Mock: Message to {to_contact} with content '{message_body}' would have been sent. (Actual implementation requires a specific messaging service integration like Twilio, Vonage, etc.)" 