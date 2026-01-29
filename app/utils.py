import requests

def get_weather(city, api_key):
    """Fetches real-time temperature and humidity using OpenWeatherMap."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        res = requests.get(url).json()
        if res.get("main"):
            return {
                "temp": res["main"]["temp"],
                "humidity": res["main"]["humidity"]
            }
    except Exception as e:
        print(f"Weather API Error: {e}")
    return None

def get_mandi_price(commodity, api_key):
    """
    Fetches the latest modal price from data.gov.in (Agmarknet).
    Your API Key is embedded in the request.
    """
    # Resource ID for 'Current Daily Price of Various Commodities from Various Markets (Mandi)'
    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
    
    # We filter by commodity name (e.g., 'Rice' or 'Potato')
    url = (
        f"https://api.data.gov.in/resource/{resource_id}?"
        f"api-key={api_key}&format=json&filters[commodity]={commodity}"
    )
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get('records') and len(data['records']) > 0:
            # We take the most recent record found
            latest_record = data['records'][0]
            price = latest_record.get('modal_price')
            mandi = latest_record.get('market')
            state = latest_record.get('state')
            return f"{price} (at {mandi}, {state})"
        else:
            return "No recent price found for this crop."
    except Exception as e:
        return f"Market Data Error: {str(e)}"