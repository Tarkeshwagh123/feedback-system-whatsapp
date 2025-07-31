from config import ALERT_RECIPIENTS

def check_for_low_ratings(citizen_contact, ref_id, rating):
    """Send alerts for low ratings"""
    if rating <= 2:  # Low rating threshold
        alert_message = f"ALERT: Low rating ({rating}/5) received for reference ID {ref_id} from {citizen_contact}. Immediate attention required."
        
        from whatsapp_bot import send_whatsapp_message
        
        # Send alert to all configured recipients
        for recipient in ALERT_RECIPIENTS:
            send_whatsapp_message(recipient, alert_message)
            
        return True
    return False