import os
import time
from pyngrok import ngrok
from dotenv import load_dotenv

def start_broadcast():
    load_dotenv()
    
    # 1. Get Token
    auth_token = "39973VNZFtcWHKPjBr1AEfdAQOS_6D4P4t6ATtcngpbMgTZsb"
    if not auth_token or "your_ngrok_token" in auth_token or auth_token == "":
        print("❌ Error: NGROK_AUTH_TOKEN is missing or invalid in your .env file.")
        print("Please visit https://dashboard.ngrok.com/get-started/your-authtoken to get your token.")
        return

    # 2. Configure Ngrok
    try:
        ngrok.set_auth_token(auth_token)
        
        # We target port 8501 which is the default for Streamlit
        # If you are running the FastAPI backend, you might want port 8000
        port = 8501 
        
        print(f"📡 Opening tunnel to port {port}...")
        public_url = ngrok.connect(port, "http")
        
        print("\n" + "="*50)
        print(f"🚀 YOUR APP IS NOW LIVE ONLINE!")
        print(f"🔗 URL: {public_url}")
        print("="*50)
        print("\nKeep this terminal open to maintain the connection.")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"⚠️ Ngrok Error: {e}")
        print("\nPossible fixes:")
        print("1. Make sure no other Ngrok process is running.")
        print("2. Check if your internet allows Ngrok (Proxy/Firewall).")
        print("3. Verify your Auth Token is correct.")

if __name__ == "__main__":
    start_broadcast()
