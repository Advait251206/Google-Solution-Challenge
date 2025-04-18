import streamlit as st
import os
import datetime
import random
import requests
import pandas as pd
from dotenv import load_dotenv
import logging
from collections import defaultdict
import io
import folium
from folium.plugins import Geocoder

try:
    from streamlit_folium import st_folium
except ImportError:
     st.error("Required libraries `folium` and `streamlit-folium` not found. Install: `pip install folium streamlit-folium`")
     st.stop()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.error("Required library `langchain-google-genai` not found. Install: `pip install langchain-google-genai pandas streamlit-folium folium python-dotenv requests gTTS`")
    LANGCHAIN_AVAILABLE = False
    st.stop()

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    st.error("Required library `gTTS` not found for audio playback. Install: `pip install gTTS`")
    GTTS_AVAILABLE = False


load_dotenv()
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/forecast"
FARMER_CSV_PATH = "Data.csv"
QA_LOG_PATH = "Log.csv"
CSV_COLUMNS = ['name', 'language', 'latitude', 'longitude', 'soil_type', 'farm_size_ha']
QA_LOG_COLUMNS = ['timestamp', 'farmer_name', 'language', 'query', 'response', 'internal_prompt']

MAP_DEFAULT_LAT = 20.5937
MAP_DEFAULT_LON = 78.9629
PROFILE_DEFAULT_LAT = 0.0
PROFILE_DEFAULT_LON = 0.0
MAP_CLICK_ZOOM = 14

SOIL_TYPES = [
    "Unknown", "Alluvial Soil", "Black Soil (Regur)", "Red Soil", "Laterite Soil",
    "Desert Soil (Arid Soil)", "Mountain Soil (Forest Soil)", "Saline Soil (Alkaline Soil)",
    "Peaty Soil (Marshy Soil)", "Loamy Soil", "Sandy Loam", "Silt Loam", "Clay Loam",
    "Sandy Clay", "Silty Clay", "Sandy Soil", "Silty Soil", "Clay Soil", "Chalky Soil", "Other"
]


TTS_LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
}


translations = {
    "English": {
        "page_title": "Krishi-Sahayak AI", "page_caption": "AI-Powered Agricultural Advice", "sidebar_config_header": "⚙️ Configuration",
        "gemini_key_label": "Google Gemini API Key", "gemini_key_help": "Required for AI responses.", "weather_key_label": "OpenWeatherMap API Key",
        "weather_key_help": "Required for weather forecasts.", "sidebar_profile_header": "👤 Farmer Profile", "farmer_name_label": "Enter Farmer Name",
        "load_profile_button": "Load Profile", "new_profile_button": "New Profile", "profile_loaded_success": "Loaded profile for {name}.",
        "profile_not_found_warning": "No profile found for '{name}'. Click 'New Profile' to create one.", "profile_exists_warning": "Profile for '{name}' already exists. Loading existing profile.",
        "creating_profile_info": "Creating new profile for '{name}'. Fill details below.", "new_profile_form_header": "New Profile for {name}",
        "pref_lang_label": "Preferred Language", "soil_type_label": "Select Soil Type",
        "location_method_label": "Set Farm Location",
        "loc_method_map": "Set Location Manually (Use Map for Reference)",
        "latitude_label": "Latitude", "longitude_label": "Longitude",
        "map_instructions": "Use map search (top-right) or click the map to find coordinates for reference. Enter them manually below.",
        "map_click_reference": "Map Click Coordinates (Reference):",
        "selected_coords_label": "Farm Coordinates (Enter Manually):",
        "farm_size_label": "Farm Size (Hectares)", "save_profile_button": "Save New Profile",
        "profile_saved_success": "Created and loaded profile for {name}.", "name_missing_error": "Farmer name cannot be empty.", "active_profile_header": "✅ Active Profile",
        "active_profile_name": "Name", "active_profile_lang": "Pref. Lang", "active_profile_loc": "Location", "active_profile_soil": "Soil", "active_profile_size": "Size (Ha)",
        "no_profile_loaded_info": "No farmer profile loaded. Enter a name and load or create.", "sidebar_output_header": "🌐 Language Settings", "select_language_label": "Select Site & Response Language",
        "tab_new_chat": "💬 New Chat", "tab_past_interactions": "📜 Past Interactions", "tab_edit_profile": "✏️ Edit Profile",
        "main_header": "Chat with Krishi-Sahayak AI", "query_label": "Enter your question:", "get_advice_button": "Send",
        "thinking_spinner": "🤖 Analyzing & Generating Advice in {lang}...",
        "advice_header": "💡 Advice for {name} (in {lang})",
        "profile_error": "❌ Please load or create a farmer profile first using the sidebar.", "query_warning": "⚠️ Please enter a question.", "gemini_key_error": "❌ Please enter your Google Gemini API Key in the sidebar.",
        "processing_error": "A critical error occurred during processing: {e}", "llm_init_error": "Could not initialize the AI model. Check the API key and try again.",
        "debug_prompt_na": "N/A",
        "intent_crop": "Farmer Query Intent: Crop Recommendation Request",
        "intent_market": "Farmer Query Intent: Market Price Inquiry",
        "intent_weather": "Farmer Query Intent: Weather Forecast & Implications Request",
        "intent_health": "Farmer Query Intent: Plant Health/Problem Diagnosis",
        "intent_general": "Farmer Query Intent: General Farming Question",
        "context_header_weather": "--- Relevant Weather Data for {location} (Interpret for Farmer) ---",
        "context_footer_weather": "--- End Weather Data ---",
        "context_weather_unavailable": "Weather Forecast Unavailable: {error_msg}",
        "context_header_crop": "--- Crop Suggestion Analysis Factors ---",
        "context_factors_crop": "Factors Considered: Soil='{soil}', Season='{season}'.",
        "context_crop_ideas": "Initial Suitable Crop Ideas: {crops}. (Analyze these based on profile/weather/market)",
        "context_footer_crop": "--- End Crop Suggestion Factors ---",
        "context_header_market": "--- Market Price Indicators for {crop} in {market} (Interpret Trend) ---",
        "context_data_market": "Forecast {days} days: Range ~₹{price_start:.2f} - ₹{price_end:.2f} / Quintal. Trend Analysis: {trend}.",
        "context_footer_market": "--- End Market Price Indicators ---",
        "context_header_health": "--- Initial Plant Health Assessment (Placeholder) ---",
        "context_data_health": "Potential Issue: '{disease}' (Confidence: {confidence:.0%}). Suggestion: {treatment}. (Please verify visually).",
        "context_footer_health": "--- End Plant Health Assessment ---",
        "context_header_general": "--- General Query Context ---",
        "context_data_general": "Farmer Question: '{query}'. (Provide a comprehensive agricultural answer based on profile/history/general knowledge.)",
        "context_footer_general": "--- End General Query Context ---",
        "crop_suggestion_data": "Crop Suggestion Data: Based on soil '{soil}' in season '{season}', consider: {crops}.",
        "market_price_data": "Market Price Data for {crop} in {market}: Expected price range (per quintal) over next {days} days: {price_start:.2f} to {price_end:.2f}. Trend: {trend}",
        "weather_data_header": "Weather Forecast Data for {location} (Next ~5 days):",
        "weather_data_error": "Weather Forecast Error: {message}",
        "plant_health_data": "Plant Health Data (Placeholder): Finding: '{disease}' ({confidence:.0%} confidence). Suggestion: {treatment}",
        "general_query_data": "Farmer Query: '{query}'. Provide a concise agricultural answer based on general knowledge.",
        "farmer_context_data": "Farmer Context: Name: {name}, Location: {location_description}, Soil: {soil}, Farm Size: {size}.",
        "session_history_header": "Current Conversation History:",
        "session_history_entry": "{role} ({lang}): {query}\n",
        "location_set_description": "Farm Near {lat:.2f},{lon:.2f}",
        "location_not_set_description": "Location Not Set",
        "past_interactions_header": "All Past Interactions for {name}",
        "log_entry_display": "<small>**Timestamp:** {timestamp}<br>**Query:** {query}<br>**Answer ({lang}):** {response}</small>\n\n---\n",
        "no_past_interactions": "No past interactions logged for this farmer.",
        "system_error_label": "System Error", "log_file_corrupt_columns": "Error: Past interactions log file ({path}) is missing expected columns: {cols}. Please check or recreate the file.",
        "error_displaying_logs": "Error reading or displaying past interactions: {error}", "profile_reload_error_after_save": "Internal error: Could not reload profile immediately after saving/updating. Please try loading it manually.",
        "db_update_error_on_save": "Internal error: Failed to update the profile database.", "map_click_invalid_coords_message": "Invalid reference coordinates stored. Click the map again.",
        "map_click_prompt_message": "Click map to get coordinates for reference.", "weather_error_summary_generation": "Could not generate daily forecast summary from the retrieved weather data.",
        "conditions_unclear": "Conditions unclear", "value_na": "N/A", "label_today": "Today", "label_tomorrow": "Tomorrow",
        "weather_rain_display": f" Rain: {{value:.1f}}mm",
        "weather_alerts_display": f". Alerts: {{alerts_joined}}",
        "weather_error_401": "Weather Forecast Error: Invalid API Key (Unauthorized). Please check the key in the sidebar.",
        "weather_error_404": "Weather Forecast Error: Location not found by the weather service.",
        "weather_error_429": "Weather Forecast Error: API rate limit exceeded. Please try again later.",
        "weather_error_http": "Weather Forecast Error: Could not fetch weather data (HTTP {status_code}).",
        "weather_error_network": "Network error connecting to weather service. Please check your internet connection.",
        "weather_error_unexpected": "An unexpected error occurred while getting or processing weather data: {error}",
        "weather_error_unknown": "Could not get weather forecast (unknown reason).",
        "your_area": "your area", "unknown_farmer": "Unknown Farmer", "not_set_label": "Not Set",
        "invalid_date_label": "Invalid Date", "no_crops_recommendation": "None specific recommended based on initial analysis.",
        "edit_profile_header": "Edit Profile for {name}", "save_changes_button": "Save Changes", "profile_updated_success": "Profile for {name} updated successfully.",
        "profile_name_edit_label": "Farmer Name (Cannot be changed)",
        "tts_button_label": "▶️ Play Audio",
        "tts_button_tooltip": "Read aloud in {lang}",
        "tts_generating_spinner": "Generating audio in {lang}...",
        "tts_error_generation": "Could not generate audio: {err}",
        "tts_error_unsupported_lang": "Audio playback not supported for {lang}",
        "tts_error_library_missing": "Audio library (gTTS) not installed.",
    },
    "Hindi": {
        "page_title": "कृषि-सहायक एआई", "page_caption": "एआई-संचालित कृषि सलाह", "sidebar_config_header": "⚙️ सेटिंग",
        "gemini_key_label": "गूगल जेमिनी एपीआई कुंजी", "gemini_key_help": "एआई प्रतिक्रियाओं के लिए आवश्यक।", "weather_key_label": "ओपनवेदरमैप एपीआई कुंजी",
        "weather_key_help": "मौसम पूर्वानुमान के लिए आवश्यक।", "sidebar_profile_header": "👤 किसान प्रोफाइल", "farmer_name_label": "किसान का नाम दर्ज करें", "load_profile_button": "प्रोफ़ाइल लोड करें",
        "new_profile_button": "नई प्रोफ़ाइल", "profile_loaded_success": "{name} के लिए प्रोफ़ाइल लोड की गई।", "profile_not_found_warning": "'{name}' के लिए कोई प्रोफ़ाइल नहीं मिली। 'नई प्रोफ़ाइल' बनाने के लिए क्लिक करें।",
        "profile_exists_warning": "'{name}' के लिए प्रोफ़ाइल पहले से मौजूद है। मौजूदा प्रोफ़ाइल लोड हो रही है।", "creating_profile_info": "'{name}' के लिए नई प्रोफ़ाइल बनाई जा रही है। नीचे विवरण भरें।",
        "new_profile_form_header": "{name} के लिए नई प्रोफ़ाइल", "pref_lang_label": "पसंदीदा भाषा", "soil_type_label": "मिट्टी का प्रकार चुनें",
        "location_method_label": "खेत का स्थान निर्धारित करें",
        "loc_method_map": "स्थान मैन्युअल रूप से सेट करें (संदर्भ के लिए मानचित्र का उपयोग करें)",
        "latitude_label": "अक्षांश", "longitude_label": "देशांतर",
        "map_instructions": "निर्देशांक संदर्भ के लिए मानचित्र खोज (ऊपर-दाईं ओर) या मानचित्र पर क्लिक करें। उन्हें नीचे मैन्युअल रूप से दर्ज करें।",
        "map_click_reference": "मानचित्र क्लिक निर्देशांक (संदर्भ):",
        "selected_coords_label": "खेत निर्देशांक (मैन्युअल रूप से दर्ज करें):",
        "farm_size_label": "खेत का आकार (हेक्टेयर)", "save_profile_button": "नई प्रोफ़ाइल सहेजें",
        "profile_saved_success": "{name} के लिए प्रोफ़ाइल बनाई और लोड की गई।", "name_missing_error": "किसान का नाम खाली नहीं हो सकता।", "active_profile_header": "✅ सक्रिय प्रोफ़ाइल",
        "active_profile_name": "नाम", "active_profile_lang": "पसंदीदा भाषा", "active_profile_loc": "स्थान", "active_profile_soil": "मिट्टी", "active_profile_size": "आकार (हेक्टेयर)",
        "no_profile_loaded_info": "कोई किसान प्रोफ़ाइल लोड नहीं हुई। नाम दर्ज करें और लोड करें या बनाएं।", "sidebar_output_header": "🌐 भाषा सेटिंग्स", "select_language_label": "साइट और प्रतिक्रिया भाषा चुनें",
        "tab_new_chat": "💬 नई चैट", "tab_past_interactions": "📜 पिछली बातचीत", "tab_edit_profile": "✏️ प्रोफ़ाइल संपादित करें",
        "main_header": "कृषि-सहाय्यक एआई के साथ चैट करें", "query_label": "अपना प्रश्न दर्ज करें:", "get_advice_button": "भेजें",
        "thinking_spinner": "🤖 विश्लेषण और {lang} में सलाह उत्पन्न हो रही है...",
        "advice_header": "💡 {name} के लिए सलाह ({lang} में)",
        "profile_error": "❌ कृपया पहले साइडबार का उपयोग करके किसान प्रोफ़ाइल लोड करें या बनाएं।", "query_warning": "⚠️ कृपया एक प्रश्न दर्ज करें।", "gemini_key_error": "❌ कृपया साइडबार में अपनी गूगल जेमिनी एपीआई कुंजी दर्ज करें।",
        "processing_error": "प्रसंस्करण के दौरान एक गंभीर त्रुटि हुई: {e}", "llm_init_error": "एआई मॉडल को इनिशियलाइज़ नहीं किया जा सका। एपीआई कुंजी जांचें और पुनः प्रयास करें।",
        "debug_prompt_na": "लागू नहीं",
        "intent_crop": "किसान प्रश्न इरादा: फसल सिफारिश अनुरोध",
        "intent_market": "किसान प्रश्न इरादा: बाजार मूल्य पूछताछ",
        "intent_weather": "किसान प्रश्न इरादा: मौसम पूर्वानुमान और प्रभाव अनुरोध",
        "intent_health": "किसान प्रश्न इरादा: पौधे का स्वास्थ्य/समस्या निदान",
        "intent_general": "किसान प्रश्न इरादा: सामान्य खेती का प्रश्न",
        "context_header_weather": "--- प्रासंगिक मौसम डेटा {location} के लिए (किसान के लिए व्याख्या करें) ---",
        "context_footer_weather": "--- मौसम डेटा समाप्त ---",
        "context_weather_unavailable": "मौसम पूर्वानुमान अनुपलब्ध: {error_msg}",
        "context_header_crop": "--- फसल सुझाव विश्लेषण कारक ---",
        "context_factors_crop": "विचाराधीन कारक: मिट्टी='{soil}', मौसम='{season}'.",
        "context_crop_ideas": "प्रारंभिक उपयुक्त फसल विचार: {crops}. (प्रोफ़ाइल/मौसम/बाजार के आधार पर इनका विश्लेषण करें)",
        "context_footer_crop": "--- फसल सुझाव कारक समाप्त ---",
        "context_header_market": "--- {market} में {crop} के लिए बाजार मूल्य संकेतक (रुझान की व्याख्या करें) ---",
        "context_data_market": "पूर्वानुमान {days} दिन: रेंज ~₹{price_start:.2f} - ₹{price_end:.2f} / क्विंटल। रुझान विश्लेषण: {trend}.",
        "context_footer_market": "--- बाजार मूल्य संकेतक समाप्त ---",
        "context_header_health": "--- प्रारंभिक पादप स्वास्थ्य मूल्यांकन (प्लेसहोल्डर) ---",
        "context_data_health": "संभावित समस्या: '{disease}' (विश्वास: {confidence:.0%})। सुझाव: {treatment}। (कृपया दृश्यात्मक रूप से सत्यापित करें)।",
        "context_footer_health": "--- पादप स्वास्थ्य मूल्यांकन समाप्त ---",
        "context_header_general": "--- सामान्य प्रश्न संदर्भ ---",
        "context_data_general": "किसान का प्रश्न: '{query}'। (प्रोफ़ाइल/इतिहास/सामान्य ज्ञान के आधार पर व्यापक कृषि उत्तर प्रदान करें।)",
        "context_footer_general": "--- सामान्य प्रश्न संदर्भ समाप्त ---",
        "crop_suggestion_data": "फसल सुझाव डेटा: '{soil}' मिट्टी और '{season}' मौसम के आधार पर, इन पर विचार करें: {crops}.",
        "market_price_data": "{crop} के लिए {market} में बाजार मूल्य डेटा: अगले {days} दिनों में अपेक्षित मूल्य सीमा (प्रति क्विंटल): {price_start:.2f} से {price_end:.2f} तक। रुझान: {trend}",
        "weather_data_header": "{location} के पास मौसम पूर्वानुमान डेटा (अगले ~5 दिन):", "weather_data_error": "मौसम पूर्वानुमान त्रुटि: {message}",
        "plant_health_data": "पौधों का स्वास्थ्य डेटा (प्लेसहोल्डर): निष्कर्ष: '{disease}' ({confidence:.0%} विश्वास)। सुझाव: {treatment}",
        "general_query_data": "किसान का प्रश्न: '{query}'. सामान्य ज्ञान के आधार पर संक्षिप्त कृषि उत्तर प्रदान करें।",
        "farmer_context_data": "किसान संदर्भ: नाम: {name}, स्थान: {location_description}, मिट्टी: {soil}, खेत का आकार: {size}.",
        "session_history_header": "वर्तमान बातचीत का इतिहास:",
        "session_history_entry": "{role} ({lang}): {query}\n",
        "location_set_description": "खेत {lat:.2f},{lon:.2f} के पास", "location_not_set_description": "स्थान निर्धारित नहीं है",
        "past_interactions_header": "{name} के लिए सभी पिछली बातचीत",
        "log_entry_display": "<small>**समय:** {timestamp}<br>**प्रश्न:** {query}<br>**उत्तर ({lang}):** {response}</small>\n\n---\n",
        "no_past_interactions": "इस किसान के लिए कोई पिछली बातचीत लॉग नहीं की गई।",
        "system_error_label": "सिस्टम त्रुटि", "log_file_corrupt_columns": "त्रुटि: पिछली बातचीत की लॉग फ़ाइल ({path}) में अपेक्षित कॉलम गायब हैं: {cols}। कृपया फ़ाइल जाँचें या पुनः बनाएँ।",
        "error_displaying_logs": "पिछली बातचीत पढ़ते या प्रदर्शित करते समय त्रुटि: {error}", "profile_reload_error_after_save": "आंतरिक त्रुटि: सहेजने/अपडेट करने के तुरंत बाद प्रोफ़ाइल पुनः लोड नहीं हो सकी। कृपया इसे मैन्युअल रूप से लोड करने का प्रयास करें।",
        "db_update_error_on_save": "आंतरिक त्रुटि: प्रोफ़ाइल डेटाबेस को अद्यतन करने में विफल।", "map_click_invalid_coords_message": "अमान्य संदर्भ निर्देशांक संग्रहीत हैं। कृपया मानचित्र पर फिर से क्लिक करें।",
        "map_click_prompt_message": "संदर्भ के लिए निर्देशांक प्राप्त करने हेतु मानचित्र पर क्लिक करें।", "weather_error_summary_generation": "प्राप्त मौसम डेटा से दैनिक पूर्वानुमान सारांश उत्पन्न नहीं किया जा सका।",
        "conditions_unclear": "स्थितियां अस्पष्ट", "value_na": "लागू नहीं", "label_today": "आज", "label_tomorrow": "कल",
        "weather_rain_display": f" बारिश: {{value:.1f}}मिमी", "weather_alerts_display": f". अलर्ट: {{alerts_joined}}",
        "weather_error_401": "मौसम पूर्वानुमान त्रुटि: अमान्य एपीआई कुंजी (अनधिकृत)। कृपया साइडबार में कुंजी जांचें।",
        "weather_error_404": "मौसम पूर्वानुमान त्रुटि: मौसम सेवा द्वारा स्थान नहीं मिला।",
        "weather_error_429": "मौसम पूर्वानुमान त्रुटि: एपीआई दर सीमा पार हो गई। कृपया बाद में पुनः प्रयास करें।",
        "weather_error_http": "मौसम पूर्वानुमान त्रुटि: मौसम डेटा प्राप्त नहीं किया जा सका (HTTP {status_code})।",
        "weather_error_network": "मौसम सेवा से कनेक्ट करने में नेटवर्क त्रुटि। कृपया अपना इंटरनेट कनेक्शन जांचें।",
        "weather_error_unexpected": "मौसम डेटा प्राप्त करते या संसाधित करते समय एक अप्रत्याशित त्रुटि हुई: {error}",
        "weather_error_unknown": "मौसम पूर्वानुमान प्राप्त नहीं किया जा सका (अज्ञात कारण)।",
        "your_area": "आपका क्षेत्र", "unknown_farmer": "अज्ञात किसान", "not_set_label": "सेट नहीं",
        "invalid_date_label": "अमान्य तारीख", "no_crops_recommendation": "प्रारंभिक विश्लेषण के आधार पर कोई विशिष्ट सुझाव नहीं दिया गया।",
        "edit_profile_header": "{name} के लिए प्रोफ़ाइल संपादित करें", "save_changes_button": "बदलाव सहेजें", "profile_updated_success": "{name} के लिए प्रोफ़ाइल सफलतापूर्वक अपडेट की गई।",
        "profile_name_edit_label": "किसान का नाम (बदला नहीं जा सकता)",
        "tts_button_label": "▶️ ऑडियो चलाएं", "tts_button_tooltip": "{lang} में जोर से पढ़ें",
        "tts_generating_spinner": "{lang} में ऑडियो बना रहा हूँ...", "tts_error_generation": "ऑडियो बनाने में विफल: {err}",
        "tts_error_unsupported_lang": "{lang} के लिए ऑडियो प्लेबैक समर्थित नहीं है", "tts_error_library_missing": "ऑडियो लाइब्रेरी (gTTS) स्थापित नहीं है।",
    },
     "Tamil": {
        "edit_profile_header": "{name} க்கான சுயவிவரத்தைத் திருத்து",
        "save_changes_button": "மாற்றங்களைச் சேமி",
        "profile_updated_success": "{name} க்கான சுயவிவரம் வெற்றிகரமாகப் புதுப்பிக்கப்பட்டது.",
        "profile_name_edit_label": "விவசாயி பெயர் (மாற்ற முடியாது)",
        "loc_method_map": "இருப்பிடத்தை கைமுறையாக அமைக்கவும் (குறிப்புக்கு வரைபடத்தைப் பயன்படுத்தவும்)",
        "map_instructions": "குறிப்புகளைக் கண்டறிய வரைபடத் தேடலைப் பயன்படுத்தவும் (மேல்-வலது) அல்லது வரைபடத்தில் கிளிக் செய்யவும். கீழே அவற்றை கைமுறையாக உள்ளிடவும்.",
        "map_click_reference": "வரைபட கிளிக் ஒருங்கிணைப்புகள் (குறிப்பு):",
        "selected_coords_label": "பண்ணை ஒருங்கிணைப்புகள் (கைமுறையாக உள்ளிடவும்):",
        "location_set_description": "பண்ணை {lat:.2f},{lon:.2f} அருகில்",
        "location_not_set_description": "இருப்பிடம் அமைக்கப்படவில்லை",
        "farmer_context_data": "விவசாயி சூழல்: பெயர்: {name}, இருப்பிடம்: {location_description}, மண்: {soil}, பண்ணை அளவு: {size}.",
        "page_caption": "AI-உந்துதல் விவசாய ஆலோசனை", "sidebar_config_header": "⚙️ கட்டமைப்பு",
        "gemini_key_label": "கூகுள் ஜெமினி API கீ", "gemini_key_help": "AI பதில்களுக்குத் தேவை.",
        "weather_key_label": "OpenWeatherMap API கீ", "weather_key_help": "வானிலை முன்னறிவிப்புகளுக்குத் தேவை.",
        "sidebar_profile_header": "👤 விவசாயி விவரக்குறிப்பு", "farmer_name_label": "விவசாயி பெயரை உள்ளிடவும்",
        "load_profile_button": "சுயவிவரத்தை ஏற்று", "new_profile_button": "புதிய சுயவிவரம்",
        "profile_loaded_success": "{name} க்கான சுயவிவரம் ஏற்றப்பட்டது.",
        "profile_not_found_warning": "'{name}' க்கான சுயவிவரம் இல்லை. புதிய ஒன்றை உருவாக்க 'புதிய சுயவிவரம்' என்பதைக் கிளிக் செய்யவும்.",
        "profile_exists_warning": "'{name}' க்கான சுயவிவரம் ஏற்கனவே உள்ளது. தற்போதுள்ள சுயவிவரத்தை ஏற்றுகிறது.",
        "creating_profile_info": "'{name}' க்கான புதிய சுயவிவரத்தை உருவாக்குகிறது. கீழே உள்ள விவரங்களை நிரப்பவும்.",
        "new_profile_form_header": "{name} க்கான புதிய சுயவிவரம்",
        "pref_lang_label": "விருப்பமான மொழி", "soil_type_label": "மண் வகையைத் தேர்ந்தெடுக்கவும்",
        "location_method_label": "பண்ணை இருப்பிடத்தை அமைக்கவும்", "latitude_label": "அட்சரேகை", "longitude_label": "தீர்க்கரேகை",
        "farm_size_label": "பண்ணை அளவு (ஹெக்டேர்)", "save_profile_button": "புதிய சுயவிவரத்தை சேமிக்கவும்",
        "profile_saved_success": "{name} க்கான சுயவிவரம் உருவாக்கப்பட்டது மற்றும் ஏற்றப்பட்டது.",
        "name_missing_error": "விவசாயி பெயர் காலியாக இருக்கக்கூடாது.",
        "active_profile_header": "✅ செயலில் உள்ள சுயவிவரம்", "active_profile_name": "பெயர்",
        "active_profile_lang": "விருப்ப. மொழி", "active_profile_loc": "இருப்பிடம்", "active_profile_soil": "மண்",
        "active_profile_size": "அளவு (Ha)",
        "no_profile_loaded_info": "விவசாயி சுயவிவரம் எதுவும் ஏற்றப்படவில்லை. பெயரை உள்ளிட்டு ஏற்றவும் அல்லது உருவாக்கவும்.",
        "sidebar_output_header": "🌐 மொழி அமைப்புகள்", "select_language_label": "தளத்தையும் மறுமொழி மொழியையும் தேர்ந்தெடுக்கவும்",
        "tab_new_chat": "💬 புதிய அரட்டை", "tab_past_interactions": "📜 கடந்த உரையாடல்கள்", "tab_edit_profile": "✏️ சுயவிவரத்தைத் திருத்து",
        "main_header": "கிருஷி-சஹாயக் AI உடன் அரட்டையடிக்கவும்", "query_label": "உங்கள் கேள்வியை உள்ளிடவும்:",
        "get_advice_button": "அனுப்பு",
        "thinking_spinner": "🤖 ஆய்வுசெய்து & {lang} மொழியில் ஆலோசனையை உருவாக்குகிறேன்...",
        "advice_header": "💡 {name} க்கான ஆலோசனை ({lang} இல்)",
        "profile_error": "❌ முதலில் பக்கப்பட்டியைப் பயன்படுத்தி விவசாயி சுயவிவரத்தை ஏற்றவும் அல்லது உருவாக்கவும்.",
        "query_warning": "⚠️ தயவுசெய்து ஒரு கேள்வியை உள்ளிடவும்.",
        "gemini_key_error": "❌ தயவுசெய்து உங்கள் கூகுள் ஜெமினி API கீயை பக்கப்பட்டியில் உள்ளிடவும்.",
        "processing_error": "செயலாக்கத்தில் ஒரு கடுமையான பிழை ஏற்பட்டது: {e}",
        "llm_init_error": "AI மாதிரியைத் தொடங்க முடியவில்லை. API கீயைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்.",
        "debug_prompt_na": "N/A", "intent_crop": "விவசாயி வினவல் நோக்கம்: பயிர் பரிந்துரை கோரிக்கை",
        "intent_market": "விவசாயி வினவல் நோக்கம்: சந்தை விலை விசாரணை",
        "intent_weather": "விவசாயி வினவல் நோக்கம்: வானிலை முன்னறிவிப்பு & தாக்கங்கள் கோரிக்கை",
        "intent_health": "விவசாயி வினவல் நோக்கம்: பயிர் சுகாதாரம்/பிரச்சனை கண்டறிதல்",
        "intent_general": "விவசாயி வினவல் நோக்கம்: பொது விவசாய கேள்வி",
        "context_header_weather": "--- Relevant Weather Data for {location} (Interpret for Farmer) ---",
        "context_footer_weather": "--- End Weather Data ---",
        "context_weather_unavailable": "Weather Forecast Unavailable: {error_msg}",
        "context_header_crop": "--- Crop Suggestion Analysis Factors ---",
        "context_factors_crop": "Factors Considered: Soil='{soil}', Season='{season}'.",
        "context_crop_ideas": "Initial Suitable Crop Ideas: {crops}. (Analyze these based on profile/weather/market)",
        "context_footer_crop": "--- End Crop Suggestion Factors ---",
        "context_header_market": "--- Market Price Indicators for {crop} in {market} (Interpret Trend) ---",
        "context_data_market": "Forecast {days} days: Range ~₹{price_start:.2f} - ₹{price_end:.2f} / Quintal. Trend Analysis: {trend}.",
        "context_footer_market": "--- End Market Price Indicators ---",
        "context_header_health": "--- Initial Plant Health Assessment (Placeholder) ---",
        "context_data_health": "Potential Issue: '{disease}' (Confidence: {confidence:.0%}). Suggestion: {treatment}. (Please verify visually).",
        "context_footer_health": "--- End Plant Health Assessment ---",
        "context_header_general": "--- General Query Context ---",
        "context_data_general": "Farmer Question: '{query}'. (Provide a comprehensive agricultural answer based on profile/history/general knowledge.)",
        "context_footer_general": "--- End General Query Context ---",
        "log_entry_display": "<small>**நேரம்:** {timestamp}<br>**கேள்வி:** {query}<br>**பதில் ({lang}):** {response}</small>\n\n---\n",
        "weather_rain_display": f" மழை: {{value:.1f}}மிமீ",
    },
    "Bengali": {
        "edit_profile_header": "{name} এর জন্য প্রোফাইল সম্পাদনা করুন",
        "save_changes_button": "পরিবর্তনগুলি সংরক্ষণ করুন",
        "profile_updated_success": "{name} এর জন্য প্রোফাইল সফলভাবে আপডেট করা হয়েছে।",
        "profile_name_edit_label": "কৃষকের নাম (পরিবর্তন করা যাবে না)",
        "loc_method_map": "অবস্থান ম্যানুয়ালি সেট করুন (রেফারেন্সের জন্য ম্যাপ ব্যবহার করুন)",
        "map_instructions": "অক্ষাংশ/দ্রাঘিমাংশ রেফারেন্সের জন্য মানচিত্র অনুসন্ধান (উপরে-ডানদিকে) ব্যবহার করুন বা মানচিত্রে ক্লিক করুন। নীচে সেগুলি ম্যানুয়ালি লিখুন।",
        "map_click_reference": "মানচিত্র ক্লিকের স্থানাঙ্ক (রেফারেন্স):",
        "selected_coords_label": "খামারের স্থানাঙ্ক (ম্যানুয়ালি লিখুন):",
        "location_set_description": "খামার {lat:.2f},{lon:.2f} এর কাছাকাছি",
        "location_not_set_description": "অবস্থান সেট করা নেই",
        "farmer_context_data": "কৃষক প্রসঙ্গ: নাম: {name}, অবস্থান: {location_description}, মাটি: {soil}, খামারের আকার: {size}.",
        "page_caption": "এআই-চালিত কৃষি পরামর্শ", "sidebar_config_header": "⚙️ কনফিগারেশন",
        "gemini_key_label": "Google Gemini API কী", "gemini_key_help": "এআই প্রতিক্রিয়ার জন্য প্রয়োজনীয়।",
        "weather_key_label": "OpenWeatherMap API কী", "weather_key_help": "আবহাওয়ার পূর্বাভাসের জন্য প্রয়োজনীয়।",
        "sidebar_profile_header": "👤 কৃষক প্রোফাইল", "farmer_name_label": "কৃষকের নাম লিখুন",
        "load_profile_button": "প্রোফাইল লোড করুন", "new_profile_button": "নতুন প্রোফাইল",
        "profile_loaded_success": "{name} এর জন্য প্রোফাইল লোড করা হয়েছে।",
        "profile_not_found_warning": "'{name}' এর জন্য কোন প্রোফাইল পাওয়া যায়নি। একটি তৈরি করতে 'নতুন প্রোফাইল' ক্লিক করুন।",
        "profile_exists_warning": "'{name}' এর প্রোফাইল ইতিমধ্যে বিদ্যমান। বিদ্যমান প্রোফাইল লোড হচ্ছে।",
        "creating_profile_info": "'{name}' এর জন্য নতুন প্রোফাইল তৈরি করা হচ্ছে। নিচে বিবরণ পূরণ করুন।",
        "new_profile_form_header": "{name} এর জন্য নতুন প্রোফাইল", "pref_lang_label": "পছন্দের ভাষা",
        "soil_type_label": "মাটির প্রকার নির্বাচন করুন", "location_method_label": "খামারের অবস্থান সেট করুন",
        "latitude_label": "অক্ষাংশ", "longitude_label": "দ্রাঘিমাংশ", "farm_size_label": "খামারের আকার (হেক্টর)",
        "save_profile_button": "নতুন প্রোফাইল সংরক্ষণ করুন",
        "profile_saved_success": "{name} এর জন্য প্রোফাইল তৈরি এবং লোড করা হয়েছে।",
        "name_missing_error": "কৃষকের নাম খালি থাকতে পারে না।", "active_profile_header": "✅ সক্রিয় প্রোফাইল",
        "active_profile_name": "নাম", "active_profile_lang": "পছন্দসই ভাষা", "active_profile_loc": "অবস্থান",
        "active_profile_soil": "মাটি", "active_profile_size": "আকার (Ha)",
        "no_profile_loaded_info": "কোন কৃষক প্রোফাইল লোড করা হয়নি। একটি নাম লিখুন এবং লোড করুন বা তৈরি করুন।",
        "sidebar_output_header": "🌐 ভাষা সেটিংস", "select_language_label": "সাইট এবং প্রতিক্রিয়া ভাষা নির্বাচন করুন",
        "tab_new_chat": "💬 নতুন চ্যাট", "tab_past_interactions": "📜 অতীত মিথস্ক্রিয়া", "tab_edit_profile": "✏️ প্রোফাইল সম্পাদনা করুন",
        "main_header": "কৃষি-সহায়ক এআই-এর সাথে চ্যাট করুন", "query_label": "আপনার প্রশ্ন লিখুন:",
        "get_advice_button": "প্রেরণ করুন",
        "thinking_spinner": "🤖 বিশ্লেষণ করছি এবং {lang} এ পরামর্শ তৈরি করছি...",
        "advice_header": "💡 {name} এর জন্য পরামর্শ ({lang} এ)",
        "profile_error": "❌ অনুগ্রহ করে সাইডবার ব্যবহার করে প্রথমে একজন কৃষকের প্রোফাইল লোড করুন বা তৈরি করুন।",
        "query_warning": "⚠️ অনুগ্রহ করে একটি প্রশ্ন লিখুন।",
        "gemini_key_error": "❌ অনুগ্রহ করে সাইডবারে আপনার Google Gemini API কী লিখুন।",
        "processing_error": "প্রসেসিং এর সময় একটি জটিল ত্রুটি ঘটেছে: {e}",
        "llm_init_error": "এআই মডেলটি চালু করা যায়নি। API কী পরীক্ষা করুন এবং আবার চেষ্টা করুন।", "debug_prompt_na": "N/A",
        "intent_crop": "কৃষকের প্রশ্নের উদ্দেশ্য: ফসল সুপারিশ অনুরোধ",
        "intent_market": "কৃষকের প্রশ্নের উদ্দেশ্য: বাজার মূল্য জিজ্ঞাসা",
        "intent_weather": "কৃষকের প্রশ্নের উদ্দেশ্য: আবহাওয়ার পূর্বাভাস এবং প্রভাব জিজ্ঞাসা",
        "intent_health": "কৃষকের প্রশ্নের উদ্দেশ্য: উদ্ভিদের স্বাস্থ্য/সমস্যা নির্ণয়",
        "intent_general": "কৃষকের প্রশ্নের উদ্দেশ্য: সাধারণ কৃষি প্রশ্ন",
        "context_header_weather": "--- Relevant Weather Data for {location} (Interpret for Farmer) ---",
        "context_footer_weather": "--- End Weather Data ---",
        "context_weather_unavailable": "Weather Forecast Unavailable: {error_msg}",
        "context_header_crop": "--- Crop Suggestion Analysis Factors ---",
        "context_factors_crop": "Factors Considered: Soil='{soil}', Season='{season}'.",
        "context_crop_ideas": "Initial Suitable Crop Ideas: {crops}. (Analyze these based on profile/weather/market)",
        "context_footer_crop": "--- End Crop Suggestion Factors ---",
        "context_header_market": "--- Market Price Indicators for {crop} in {market} (Interpret Trend) ---",
        "context_data_market": "Forecast {days} days: Range ~₹{price_start:.2f} - ₹{price_end:.2f} / Quintal. Trend Analysis: {trend}.",
        "context_footer_market": "--- End Market Price Indicators ---",
        "context_header_health": "--- Initial Plant Health Assessment (Placeholder) ---",
        "context_data_health": "Potential Issue: '{disease}' (Confidence: {confidence:.0%}). Suggestion: {treatment}. (Please verify visually).",
        "context_footer_health": "--- End Plant Health Assessment ---",
        "context_header_general": "--- General Query Context ---",
        "context_data_general": "Farmer Question: '{query}'. (Provide a comprehensive agricultural answer based on profile/history/general knowledge.)",
        "context_footer_general": "--- End General Query Context ---",
        "log_entry_display": "<small>**সময়:** {timestamp}<br>**প্রশ্ন:** {query}<br>**উত্তর ({lang}):** {response}</small>\n\n---\n",
        "weather_rain_display": f" বৃষ্টি: {{value:.1f}}মিমি",
    },
    "Telugu": {
        "edit_profile_header": "{name} కోసం ప్రొఫైల్‌ని సవరించండి",
        "save_changes_button": "మార్పులను సేవ్ చేయండి",
        "profile_updated_success": "{name} కోసం ప్రొఫైల్ విజయవంతంగా నవీకరించబడింది.",
        "profile_name_edit_label": "రైతు పేరు (మార్చబడదు)",
        "loc_method_map": "స్థానాన్ని మాన్యువల్‌గా సెట్ చేయండి (రిఫరెన్స్ కోసం మ్యాప్‌ని ఉపయోగించండి)",
        "map_instructions": "రిఫరెన్స్ కోఆర్డినేట్‌లను కనుగొనడానికి మ్యాప్ శోధన (ఎగువ-కుడి) ఉపయోగించండి లేదా మ్యాప్‌పై క్లిక్ చేయండి. వాటిని క్రింద మాన్యువల్‌గా నమోదు చేయండి.",
        "map_click_reference": "మ్యాప్ క్లిక్ కోఆర్డినేట్‌లు (రిఫరెన్స్):",
        "selected_coords_label": "వ్యవసాయ క్షేత్రం కోఆర్డినేట్‌లు (మాన్యువల్‌గా నమోదు చేయండి):",
        "location_set_description": "పొలం {lat:.2f},{lon:.2f} సమీపంలో",
        "location_not_set_description": "స్థానం సెట్ చేయబడలేదు",
        "farmer_context_data": "రైతు సందర్భం: పేరు: {name}, స్థానం: {location_description}, నేల: {soil}, క్షేత్ర పరిమాణం: {size}.",
        "page_caption": "AI- ఆధారిత వ్యవసాయ సలహా", "sidebar_config_header": "⚙️ కాన్ఫిగరేషన్",
        "gemini_key_label": "Google Gemini API కీ", "gemini_key_help": "AI ప్రతిస్పందనలకు అవసరం.",
        "weather_key_label": "OpenWeatherMap API కీ", "weather_key_help": "వాతావరణ సూచనలకు అవసరం.",
        "sidebar_profile_header": "👤 రైతు ప్రొఫైల్", "farmer_name_label": "రైతు పేరు నమోదు చేయండి",
        "load_profile_button": "ప్రొఫైల్ లోడ్ చేయండి", "new_profile_button": "కొత్త ప్రొఫైల్",
        "profile_loaded_success": "{name} కోసం ప్రొఫైల్ లోడ్ చేయబడింది.",
        "profile_not_found_warning": "'{name}' కోసం ప్రొఫైల్ కనుగొనబడలేదు. కొత్తది సృష్టించడానికి 'కొత్త ప్రొఫైల్' క్లిక్ చేయండి.",
        "profile_exists_warning": "'{name}' కోసం ప్రొఫైల్ ఇప్పటికే ఉంది. ఇప్పటికే ఉన్న ప్రొఫైల్ లోడ్ అవుతోంది.",
        "creating_profile_info": "'{name}' కోసం కొత్త ప్రొఫైల్ సృష్టిస్తోంది. క్రింద వివరాలను పూరించండి.",
        "new_profile_form_header": "{name} కోసం కొత్త ప్రొఫైల్", "pref_lang_label": "ఇష్టపడే భాష",
        "soil_type_label": "నేల రకాన్ని ఎంచుకోండి", "location_method_label": "వ్యవసాయ క్షేత్ర స్థానాన్ని సెట్ చేయండి",
        "latitude_label": "అక్షాంశం", "longitude_label": "రేఖాంశం",
        "farm_size_label": "వ్యవసాయ క్షేత్ర పరిమాణం (హెక్టార్లు)", "save_profile_button": "కొత్త ప్రొఫైల్‌ను సేవ్ చేయండి",
        "profile_saved_success": "{name} కోసం ప్రొఫైల్ సృష్టించబడింది మరియు లోడ్ చేయబడింది.",
        "name_missing_error": "రైతు పేరు ఖాళీగా ఉండకూడదు.", "active_profile_header": "✅ క్రియాశీల ప్రొఫైల్",
        "active_profile_name": "పేరు", "active_profile_lang": "ప్రాధాన్య భాష", "active_profile_loc": "స్థానం",
        "active_profile_soil": "నేల", "active_profile_size": "పరిమాణం (Ha)",
        "no_profile_loaded_info": "రైతు ప్రొఫైల్ లోడ్ కాలేదు. పేరును నమోదు చేసి లోడ్ చేయండి లేదా సృష్టించండి.",
        "sidebar_output_header": "🌐 భాషా సెట్టింగ్‌లు", "select_language_label": "సైట్ & ప్రతిస్పందన భాషను ఎంచుకోండి",
        "tab_new_chat": "💬 కొత్త చాట్", "tab_past_interactions": "📜 గత సంభాషణలు", "tab_edit_profile": "✏️ ప్రొఫైల్‌ని సవరించండి",
        "main_header": "కృషి-సహాయక్ AI తో చాట్ చేయండి", "query_label": "మీ ప్రశ్నను నమోదు చేయండి:",
        "get_advice_button": "పంపండి",
        "thinking_spinner": "🤖 విశ్లేషిస్తున్నాను & {lang} లో సలహాను ఉత్పత్తి చేస్తున్నాను...",
        "advice_header": "💡 {name} కోసం సలహా ({lang} లో)",
        "profile_error": "❌ దయచేసి ముందుగా సైడ్‌బార్‌ని ఉపయోగించి రైతు ప్రొఫైల్‌ను లోడ్ చేయండి లేదా సృష్టించండి.",
        "query_warning": "⚠️ దయచేసి ఒక ప్రశ్నను నమోదు చేయండి.",
        "gemini_key_error": "❌ దయచేసి సైడ్‌బార్‌లో మీ Google Gemini API కీని నమోదు చేయండి.",
        "processing_error": "ప్రాసెసింగ్ సమయంలో తీవ్రమైన లోపం సంభవించింది: {e}",
        "llm_init_error": "AI నమూనాని ప్రారంభించలేకపోయింది. API కీని తనిఖీ చేసి, మళ్లీ ప్రయత్నించండి.",
        "debug_prompt_na": "N/A", "intent_crop": "రైతు ప్రశ్న ఉద్దేశ్యం: పంట సిఫార్సు అభ్యర్థన",
        "intent_market": "రైతు ప్రశ్న ఉద్దేశ్యం: మార్కెట్ ధర విచారణ",
        "intent_weather": "రైతు ప్రశ్న ఉద్దేశ్యం: వాతావరణ సూచన & ప్రభావాల అభ్యర్థన",
        "intent_health": "రైతు ప్రశ్న ఉద్దేశ్యం: మొక్క ఆరోగ్య/సమస్య నిర్ధారణ",
        "intent_general": "రైతు ప్రశ్న ఉద్దేశ్యం: సాధారణ వ్యవసాయ ప్రశ్న",
        "context_header_weather": "--- Relevant Weather Data for {location} (Interpret for Farmer) ---",
        "context_footer_weather": "--- End Weather Data ---",
        "context_weather_unavailable": "Weather Forecast Unavailable: {error_msg}",
        "context_header_crop": "--- Crop Suggestion Analysis Factors ---",
        "context_factors_crop": "Factors Considered: Soil='{soil}', Season='{season}'.",
        "context_crop_ideas": "Initial Suitable Crop Ideas: {crops}. (Analyze these based on profile/weather/market)",
        "context_footer_crop": "--- End Crop Suggestion Factors ---",
        "context_header_market": "--- Market Price Indicators for {crop} in {market} (Interpret Trend) ---",
        "context_data_market": "Forecast {days} days: Range ~₹{price_start:.2f} - ₹{price_end:.2f} / Quintal. Trend Analysis: {trend}.",
        "context_footer_market": "--- End Market Price Indicators ---",
        "context_header_health": "--- Initial Plant Health Assessment (Placeholder) ---",
        "context_data_health": "Potential Issue: '{disease}' (Confidence: {confidence:.0%}). Suggestion: {treatment}. (Please verify visually).",
        "context_footer_health": "--- End Plant Health Assessment ---",
        "context_header_general": "--- General Query Context ---",
        "context_data_general": "Farmer Question: '{query}'. (Provide a comprehensive agricultural answer based on profile/history/general knowledge.)",
        "context_footer_general": "--- End General Query Context ---",
        "log_entry_display": "<small>**సమయం:** {timestamp}<br>**ప్రశ్న:** {query}<br>**సమాధానం ({lang}):** {response}</small>\n\n---\n",
        "weather_rain_display": f" వర్షం: {{value:.1f}}మిమీ",
    },
    "Marathi": {
        "edit_profile_header": "{name} साठी प्रोफाइल संपादित करा",
        "save_changes_button": "बदल जतन करा",
        "profile_updated_success": "{name} साठी प्रोफाइल यशस्वीरित्या अद्यतनित केले.",
        "profile_name_edit_label": "शेतकऱ्याचे नाव (बदलता येणार नाही)",
        "loc_method_map": "स्थान मॅन्युअली सेट करा (संदर्भासाठी नकाशा वापरा)",
        "map_instructions": "निर्देशांक संदर्भासाठी नकाशा शोध (वर-उजवीकडे) वापरा किंवा नकाशावर क्लिक करा. ते खाली मॅन्युअली प्रविष्ट करा.",
        "map_click_reference": "नकाशा क्लिक निर्देशांक (संदर्भ):",
        "selected_coords_label": "शेती निर्देशांक (मॅन्युअली प्रविष्ट करा):",
        "location_set_description": "शेत {lat:.2f},{lon:.2f} जवळ",
        "location_not_set_description": "स्थान सेट नाही",
        "farmer_context_data": "शेतकरी संदर्भ: नाव: {name}, स्थान: {location_description}, माती: {soil}, शेतीचा आकार: {size}.",
        "page_caption": "एआय-आधारित कृषी सल्ला", "sidebar_config_header": "⚙️ संरचना",
        "gemini_key_label": "गूगल जेमिनी एपीआय की", "gemini_key_help": "एआय प्रतिसादांसाठी आवश्यक.",
        "weather_key_label": "ओपनवेदरमॅप एपीआय की", "weather_key_help": "हवामान अंदाजासाठी आवश्यक.",
        "sidebar_profile_header": "👤 शेतकरी प्रोफाइल", "farmer_name_label": "शेतकऱ्याचे नाव प्रविष्ट करा",
        "load_profile_button": "प्रोफाइल लोड करा", "new_profile_button": "नवीन प्रोफाइल",
        "profile_loaded_success": "{name} साठी प्रोफाइल लोड केले.",
        "profile_not_found_warning": "'{name}' साठी कोणतेही प्रोफाइल आढळले नाही. तयार करण्यासाठी 'नवीन प्रोफाइल' क्लिक करा.",
        "profile_exists_warning": "'{name}' साठी प्रोफाइल आधीपासूनच अस्तित्वात आहे. विद्यमान प्रोफाइल लोड करत आहे.",
        "creating_profile_info": "'{name}' साठी नवीन प्रोफाइल तयार करत आहे. खाली तपशील भरा.",
        "new_profile_form_header": "{name} साठी नवीन प्रोफाइल", "pref_lang_label": "पसंतीची भाषा",
        "soil_type_label": "मातीचा प्रकार निवडा", "location_method_label": "शेतीचे स्थान सेट करा",
        "latitude_label": "अक्षांश", "longitude_label": "रेखांश", "farm_size_label": "शेतीचा आकार (हेक्टर)",
        "save_profile_button": "नवीन प्रोफाइल जतन करा",
        "profile_saved_success": "{name} साठी प्रोफाइल तयार केले आणि लोड केले.",
        "name_missing_error": "शेतकऱ्याचे नाव रिक्त असू शकत नाही.", "active_profile_header": "✅ सक्रिय प्रोफाइल",
        "active_profile_name": "नाव", "active_profile_lang": "पसंतीची भाषा", "active_profile_loc": "स्थान",
        "active_profile_soil": "माती", "active_profile_size": "आकार (हेक्टर)",
        "no_profile_loaded_info": "शेतकरी प्रोफाइल लोड केलेले नाही. नाव प्रविष्ट करा आणि लोड करा किंवा तयार करा.",
        "sidebar_output_header": "🌐 भाषा सेटिंग्ज", "select_language_label": "साइट आणि प्रतिसाद भाषा निवडा",
        "tab_new_chat": "💬 नवीन चॅट", "tab_past_interactions": "📜 मागील संवाद", "tab_edit_profile": "✏️ प्रोफाइल संपादित करा",
        "main_header": "कृषी-सहाय्यक एआय सह चॅट करा", "query_label": "आपला प्रश्न प्रविष्ट करा:",
        "get_advice_button": "पाठवा",
        "thinking_spinner": "🤖 विश्लेषण करत आहे आणि {lang} मध्ये सल्ला तयार करत आहे...",
        "advice_header": "💡 {name} साठी सल्ला ({lang} मध्ये)",
        "profile_error": "❌ कृपया आधी साइडबार वापरून शेतकरी प्रोफाइल लोड करा किंवा तयार करा.",
        "query_warning": "⚠️ कृपया एक प्रश्न प्रविष्ट करा.",
        "gemini_key_error": "❌ कृपया साइडबारमध्ये आपला गूगल जेमिनी एपीआय की प्रविष्ट करा.",
        "processing_error": "प्रक्रियेदरम्यान एक गंभीर त्रुटी आली: {e}",
        "llm_init_error": "एआय मॉडेल सुरू करता आले नाही. एपीआय की तपासा आणि पुन्हा प्रयत्न करा.",
        "debug_prompt_na": "लागू नाही", "intent_crop": "शेतकरी क्वेरी उद्देश: पीक शिफारस विनंती",
        "intent_market": "शेतकरी क्वेरी उद्देश: बाजारभाव चौकशी",
        "intent_weather": "शेतकरी क्वेरी उद्देश: हवामान अंदाज आणि परिणाम विनंती",
        "intent_health": "शेतकरी क्वेरी उद्देश: वनस्पती आरोग्य/समस्या निदान",
        "intent_general": "शेतकरी क्वेरी उद्देश: सामान्य शेती प्रश्न",
        "context_header_weather": "--- Relevant Weather Data for {location} (Interpret for Farmer) ---",
        "context_footer_weather": "--- End Weather Data ---",
        "context_weather_unavailable": "Weather Forecast Unavailable: {error_msg}",
        "context_header_crop": "--- Crop Suggestion Analysis Factors ---",
        "context_factors_crop": "Factors Considered: Soil='{soil}', Season='{season}'.",
        "context_crop_ideas": "Initial Suitable Crop Ideas: {crops}. (Analyze these based on profile/weather/market)",
        "context_footer_crop": "--- End Crop Suggestion Factors ---",
        "context_header_market": "--- Market Price Indicators for {crop} in {market} (Interpret Trend) ---",
        "context_data_market": "Forecast {days} days: Range ~₹{price_start:.2f} - ₹{price_end:.2f} / Quintal. Trend Analysis: {trend}.",
        "context_footer_market": "--- End Market Price Indicators ---",
        "context_header_health": "--- Initial Plant Health Assessment (Placeholder) ---",
        "context_data_health": "Potential Issue: '{disease}' (Confidence: {confidence:.0%}). Suggestion: {treatment}. (Please verify visually).",
        "context_footer_health": "--- End Plant Health Assessment ---",
        "context_header_general": "--- General Query Context ---",
        "context_data_general": "Farmer Question: '{query}'. (Provide a comprehensive agricultural answer based on profile/history/general knowledge.)",
        "context_footer_general": "--- End General Query Context ---",
        "log_entry_display": "<small>**वेळ:** {timestamp}<br>**प्रश्न:** {query}<br>**उत्तर ({lang}):** {response}</small>\n\n---\n",
        "weather_rain_display": f" पाऊस: {{value:.1f}}मिमी",
    },

}


def _format_translation(template, **kwargs):
    formatted_kwargs = {}
    for k, v in kwargs.items():
        if pd.isna(v):
             formatted_kwargs[k] = ui_translator("value_na", default="N/A")
        elif isinstance(v, float):
             if k in ['price_start', 'price_end', 'farm_size_ha']: formatted_kwargs[k] = f"{v:.2f}"
             elif k in ['latitude', 'longitude']: formatted_kwargs[k] = f"{v:.6f}"
             elif k == 'confidence': formatted_kwargs[k] = f"{v:.0%}"
             elif k == 'value':
                 formatted_kwargs[k] = f"{v:.1f}"
             else: formatted_kwargs[k] = f"{v}"
        elif isinstance(v, (int, datetime.date, datetime.datetime)):
             formatted_kwargs[k] = v
        elif v is None:
            formatted_kwargs[k] = ""
        else:
            formatted_kwargs[k] = str(v)

    try:
        str_template = str(template)
        temp_template = str_template.replace('{{', '<DOUBLE_BRACE_OPEN>').replace('}}', '<DOUBLE_BRACE_CLOSE>')
        formatted = temp_template.format(**formatted_kwargs)
        formatted = formatted.replace('<DOUBLE_BRACE_OPEN>', '{{').replace('<DOUBLE_BRACE_CLOSE>', '}}')
        return formatted
    except KeyError as e:
        logger.warning(f"Translator: Missing format key '{e}' in template. Template: '{template}' Kwargs: {kwargs}")
        return template
    except ValueError as e:
        key_causing_error = None
        for key_check in formatted_kwargs:
            if f"{{{key_check}:" in str(template):
                key_causing_error = key_check
                break
        if "Unknown format code" in str(e):
             logger.warning(f"Translator: Formatting error for key '{key_causing_error or 'unknown'}'. Value type: {type(kwargs.get(key_causing_error))}. Template: '{template}'")
             return template
        else:
            logger.error(f"Translator: Unexpected format value error with args {formatted_kwargs}: {e}. Template: '{template}'", exc_info=False)
            return template
    except Exception as e:
        logger.error(f"Translator: Unexpected format error with args {formatted_kwargs}: {e}. Template: '{template}'", exc_info=False)
        return template

def ui_translator(key, default=None, **kwargs):
    selected_language = st.session_state.get('selected_language', "English")

    if selected_language not in translations:
        if selected_language != "English":
            logger.warning(f"Selected language '{selected_language}' not found in translations. Falling back to English.")
            selected_language = "English"
            st.session_state.selected_language = "English"

    lang_dict = translations.get(selected_language, translations["English"])
    default_lang_dict = translations.get("English", {})

    template = lang_dict.get(key)
    if template is None:
        template = default_lang_dict.get(key)
        if template is None:
            missing_key_msg = f"[{key} NOT FOUND in {selected_language} or English]"
            logger.debug(f"Translation key '{key}' not found for language '{selected_language}' or fallback 'English'.")
            template = default if default is not None else missing_key_msg

    return _format_translation(template, **kwargs)


def load_or_create_farmer_db():
    if os.path.exists(FARMER_CSV_PATH):
        try:
            df = pd.read_csv(FARMER_CSV_PATH, encoding='utf-8')
            logger.debug(f"Read {len(df)} rows from {FARMER_CSV_PATH}")
            missing_cols = False
            for col in CSV_COLUMNS:
                if col not in df.columns:
                    missing_cols = True
                    logger.warning(f"Column '{col}' missing in {FARMER_CSV_PATH}, adding with default.")
                    if col == 'latitude': df[col] = PROFILE_DEFAULT_LAT
                    elif col == 'longitude': df[col] = PROFILE_DEFAULT_LON
                    elif col == 'farm_size_ha': df[col] = 1.0
                    elif col == 'soil_type': df[col] = 'Unknown'
                    elif col == 'language': df[col] = 'English'
                    elif col == 'name': df[col] = ''
                    else: df[col] = pd.NA

            df['name'] = df['name'].fillna('').astype(str).str.strip()
            df = df[df['name'] != '']

            df['language'] = df['language'].fillna('English').astype(str).str.strip()
            df['language'] = df['language'].apply(lambda x: x if x in translations else 'English')

            df['soil_type'] = df['soil_type'].fillna('Unknown').astype(str).str.strip()

            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(PROFILE_DEFAULT_LAT)
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(PROFILE_DEFAULT_LON)
            df['farm_size_ha'] = pd.to_numeric(df['farm_size_ha'], errors='coerce').fillna(1.0)
            df['farm_size_ha'] = df['farm_size_ha'].apply(lambda x: x if pd.notna(x) and x > 0 else 1.0)

            df = df[CSV_COLUMNS]

            if missing_cols:
                logger.info(f"Resaving {FARMER_CSV_PATH} after adding missing columns.")
                try:
                    save_farmer_db(df)
                except Exception as save_err:
                    logger.error(f"Failed to resave {FARMER_CSV_PATH} after fixing columns: {save_err}")
                    st.warning(f"Could not auto-correct {FARMER_CSV_PATH}. Please check file integrity.")

            logger.info(f"Loaded and validated {len(df)} profiles from {FARMER_CSV_PATH}")
            return df

        except pd.errors.EmptyDataError:
            logger.warning(f"{FARMER_CSV_PATH} is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=CSV_COLUMNS)
        except Exception as e:
            logger.error(f"Error loading or processing {FARMER_CSV_PATH}: {e}", exc_info=True)
            st.error(f"Could not load farmer profiles due to file error: {e}")
            return pd.DataFrame(columns=CSV_COLUMNS)
    else:
        logger.info(f"{FARMER_CSV_PATH} not found. Creating an empty DataFrame structure.")
        return pd.DataFrame(columns=CSV_COLUMNS)


def add_or_update_farmer(df, profile_data):
    if not isinstance(df, pd.DataFrame):
        logger.error("add_or_update_farmer received non-DataFrame.")
        return pd.DataFrame(columns=CSV_COLUMNS)

    profile_name_clean = str(profile_data.get('name', '')).strip()
    if not profile_name_clean:
        logger.warning("Attempted to add/update farmer with empty name.")
        return df

    name_lower = profile_name_clean.lower()
    if 'name' not in df.columns: df['name'] = ''
    df['name'] = df['name'].astype(str)

    existing_indices = df.index[df['name'].str.lower() == name_lower].tolist()

    new_data = {}
    for col in CSV_COLUMNS:
        value = profile_data.get(col)
        if col == 'latitude':
            default_val = PROFILE_DEFAULT_LAT
            num_val = pd.to_numeric(value, errors='coerce')
            final_val = default_val if pd.isna(num_val) else float(num_val)
            new_data[col] = final_val
            if pd.isna(num_val) and value is not None and str(value).strip() != "":
                logger.warning(f"Invalid value '{value}' provided for {col} for farmer '{profile_name_clean}'. Using default {default_val}.")
        elif col == 'longitude':
            default_val = PROFILE_DEFAULT_LON
            num_val = pd.to_numeric(value, errors='coerce')
            final_val = default_val if pd.isna(num_val) else float(num_val)
            new_data[col] = final_val
            if pd.isna(num_val) and value is not None and str(value).strip() != "":
                logger.warning(f"Invalid value '{value}' provided for {col} for farmer '{profile_name_clean}'. Using default {default_val}.")
        elif col == 'farm_size_ha':
            default_val = 1.0
            num_val = pd.to_numeric(value, errors='coerce')
            value_float = default_val if pd.isna(num_val) else float(num_val)
            final_val = value_float if value_float > 0 else default_val
            new_data[col] = final_val
            if pd.isna(num_val) and value is not None and str(value).strip() != "":
                 logger.warning(f"Invalid value '{value}' provided for {col} for farmer '{profile_name_clean}'. Using default {default_val}.")
            elif value_float <= 0 and value is not None:
                 logger.warning(f"Non-positive value '{value}' provided for {col} for farmer '{profile_name_clean}'. Using default {default_val}.")
        elif col == 'name':
             new_data[col] = profile_name_clean
        elif col == 'language':
            cleaned_value = str(value).strip() if pd.notna(value) else ''
            new_data[col] = cleaned_value if cleaned_value in translations else 'English'
        elif col == 'soil_type':
             cleaned_value = str(value).strip() if pd.notna(value) else ''
             new_data[col] = cleaned_value if cleaned_value else 'Unknown'
        else:
             new_data[col] = str(value).strip() if pd.notna(value) else ''

    logger.debug(f"add_or_update_farmer: Prepared validated data for {profile_name_clean}: {new_data}")

    if not new_data.get('name'):
        logger.error(f"Farmer name became invalid after cleaning for data: {profile_data}")
        return df

    if existing_indices:
        idx_to_update = existing_indices[0]
        logger.info(f"Updating profile for '{profile_name_clean}' at index {idx_to_update}")
        try:
            for col_assign in CSV_COLUMNS:
                if col_assign not in df.columns: df[col_assign] = None
            for col_name in CSV_COLUMNS:
                 df.loc[idx_to_update, col_name] = new_data[col_name]
        except Exception as e:
            logger.error(f"Error updating DataFrame row at index {idx_to_update}: {e}", exc_info=True)
            st.error(f"Internal error updating profile for {profile_name_clean}")
            return df
        return df
    else:
        logger.info(f"Adding new profile for '{profile_name_clean}'")
        try:
            new_df_row = pd.DataFrame([new_data], columns=CSV_COLUMNS)
            df_updated = pd.concat([df, new_df_row], ignore_index=True)
            return df_updated[CSV_COLUMNS]
        except Exception as e:
            logger.error(f"Error concatenating new profile row: {e}", exc_info=True)
            st.error(f"Internal error adding profile for {profile_name_clean}")
            return df


def save_farmer_db(df):
    if not isinstance(df, pd.DataFrame):
        logger.error("Attempted to save a non-DataFrame object as farmer DB.")
        st.error("Internal error: Cannot save profile database.")
        return

    try:
        if not all(c in df.columns for c in CSV_COLUMNS):
            logger.warning(f"DataFrame missing required columns before save. Has: {df.columns.tolist()}. Reindexing.")
            df_to_save = df.reindex(columns=CSV_COLUMNS).copy()
            df_to_save['latitude'] = df_to_save['latitude'].fillna(PROFILE_DEFAULT_LAT)
            df_to_save['longitude'] = df_to_save['longitude'].fillna(PROFILE_DEFAULT_LON)
            df_to_save['farm_size_ha'] = df_to_save['farm_size_ha'].fillna(1.0)
            df_to_save['language'] = df_to_save['language'].apply(lambda x: x if pd.notna(x) and x in translations else 'English')
            df_to_save['soil_type'] = df_to_save['soil_type'].fillna('Unknown')
            df_to_save['name'] = df_to_save['name'].fillna('')
        else:
            df_to_save = df[CSV_COLUMNS].copy()

        df_to_save['name'] = df_to_save['name'].fillna('').astype(str).str.strip()
        df_to_save = df_to_save[df_to_save['name'] != '']

        df_to_save['language'] = df_to_save['language'].fillna('English').astype(str).str.strip()
        df_to_save['language'] = df_to_save['language'].apply(lambda x: x if x in translations else 'English')
        df_to_save['soil_type'] = df_to_save['soil_type'].fillna('Unknown').astype(str).str.strip()

        df_to_save['latitude'] = pd.to_numeric(df_to_save['latitude'], errors='coerce').fillna(PROFILE_DEFAULT_LAT)
        df_to_save['longitude'] = pd.to_numeric(df_to_save['longitude'], errors='coerce').fillna(PROFILE_DEFAULT_LON)
        df_to_save['farm_size_ha'] = pd.to_numeric(df_to_save['farm_size_ha'], errors='coerce').fillna(1.0)
        df_to_save['farm_size_ha'] = df_to_save['farm_size_ha'].apply(lambda x: x if pd.notna(x) and x > 0 else 1.0)

        logger.debug(f"save_farmer_db: Dataframe state just before sorting and saving ({len(df_to_save)} rows):\n{df_to_save.head().to_string()}")
        df_sorted = df_to_save.sort_values(by='name', key=lambda col: col.str.lower(), na_position='last')

        df_sorted.to_csv(FARMER_CSV_PATH, index=False, encoding='utf-8')
        logger.info(f"Successfully saved {len(df_sorted)} profiles to {FARMER_CSV_PATH}.")

    except Exception as e:
        logger.error(f"Error saving farmer profiles to {FARMER_CSV_PATH}: {e}", exc_info=True)
        st.error(f"Could not save farmer profiles: {e}")


def find_farmer(df, name):
    if df is None or df.empty or not isinstance(name, str):
        return None
    name_clean = name.strip()
    if not name_clean:
        return None

    name_lower = name_clean.lower()

    if 'name' not in df.columns:
         logger.warning("'name' column missing in DataFrame during find_farmer.")
         return None
    df['name'] = df['name'].astype(str)

    match = df.loc[df['name'].fillna('').str.lower() == name_lower]

    if not match.empty:
        profile_dict = match.iloc[0].to_dict()
        validated_profile = {}
        for col in CSV_COLUMNS:
             value = profile_dict.get(col)
             if col == 'latitude':
                 num_val = pd.to_numeric(value, errors='coerce')
                 validated_profile[col] = float(num_val) if pd.notna(num_val) else PROFILE_DEFAULT_LAT
             elif col == 'longitude':
                 num_val = pd.to_numeric(value, errors='coerce')
                 validated_profile[col] = float(num_val) if pd.notna(num_val) else PROFILE_DEFAULT_LON
             elif col == 'farm_size_ha':
                 num_val = pd.to_numeric(value, errors='coerce')
                 validated_profile[col] = float(num_val) if pd.notna(num_val) and num_val > 0 else 1.0
             elif col == 'name':
                  validated_profile[col] = str(value).strip()
             elif col == 'language':
                  lang_val = str(value).strip()
                  validated_profile[col] = lang_val if lang_val in translations else 'English'
             elif col == 'soil_type':
                  soil_val = str(value).strip()
                  validated_profile[col] = soil_val if soil_val else 'Unknown'
             else:
                 validated_profile[col] = value
        return validated_profile
    return None


def log_qa(timestamp, farmer_name, language, query, response, internal_prompt):
    try:
        log_entry = {
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'farmer_name': str(farmer_name).strip(),
            'language': str(language),
            'query': str(query),
            'response': str(response),
            'internal_prompt': str(internal_prompt)
        }
        log_df_entry = pd.DataFrame([log_entry], columns=QA_LOG_COLUMNS)
        file_exists = os.path.exists(QA_LOG_PATH)
        log_df_entry.to_csv(
            QA_LOG_PATH,
            mode='a',
            header=not file_exists,
            index=False,
            encoding='utf-8'
        )
        logger.info(f"Logged Q&A for farmer '{farmer_name}' to {QA_LOG_PATH}")
    except IOError as e:
        logger.error(f"IOError logging Q&A to {QA_LOG_PATH}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error logging Q&A to {QA_LOG_PATH}: {e}", exc_info=True)


def initialize_llm(api_key):
    if not LANGCHAIN_AVAILABLE:
        st.error("Langchain Google GenAI library not available. Cannot initialize LLM.")
        return None
    if not api_key:
        logger.warning("Attempting to initialize LLM without an API key.")
        st.error(ui_translator("gemini_key_error"))
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        logger.info("Google Gemini LLM object initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"LLM Initialization failed: {e}", exc_info=True)
        error_message = ui_translator("llm_init_error")
        err_str = str(e).lower()
        if "api_key" in err_str or "permission" in err_str or "denied" in err_str or "authenticate" in err_str:
            error_message = f"{ui_translator('llm_init_error')} " + ui_translator("gemini_key_error")
        elif "quota" in err_str or "resource has been exhausted" in err_str:
            error_message = f"{ui_translator('llm_init_error')} API quota exceeded. ({e})"
        elif "could not resolve model" in err_str:
             error_message = f"{ui_translator('llm_init_error')} Invalid model name specified. ({e})"

        st.error(error_message)
        return None


def predict_suitable_crops(soil_type, region, avg_temp, avg_rainfall, season):
    logger.debug(f"Predicting crops: Soil={soil_type}, Region={region}, Temp={avg_temp}, Rain={avg_rainfall}, Season={season}")
    recommendations = []; soil_lower = soil_type.lower() if isinstance(soil_type, str) else ""
    if "loamy" in soil_lower or "alluvial" in soil_lower:
        if avg_rainfall > 600 and season == "Kharif": recommendations.extend(["Rice", "Cotton", "Sugarcane", "Maize"])
        elif season == "Rabi": recommendations.extend(["Wheat", "Mustard", "Barley", "Gram"])
        else: recommendations.extend(["Vegetables", "Pulses"])
    elif "clay" in soil_lower or "black" in soil_lower:
         if avg_rainfall > 500 and season == "Kharif": recommendations.extend(["Cotton", "Soybean", "Sorghum", "Pigeon Pea"])
         elif season == "Rabi": recommendations.extend(["Wheat", "Gram", "Linseed"])
         else: recommendations.extend(["Pulses", "Sunflower"])
    elif "sandy" in soil_lower or "desert" in soil_lower or "arid" in soil_lower:
        if avg_temp > 25: recommendations.extend(["Bajra", "Groundnut", "Millet", "Guar"])
        else: recommendations.extend(["Mustard", "Barley", "Chickpea"])
    elif "red" in soil_lower or "laterite" in soil_lower:
         recommendations.extend(["Groundnut", "Pulses", "Potato", "Ragi", "Millets"])
    else:
        recommendations.extend(["Sorghum", "Local Pulses", "Regional Vegetables", "Fodder Crops"])
    random.shuffle(recommendations); return list(set(recommendations[:3]))

def predict_disease_from_image_placeholder():
    logger.debug("Predicting disease (placeholder function).")
    possible_results = [
        {"disease": "Healthy", "confidence": 0.95, "treatment": "No action needed."},
        {"disease": "Maize Common Rust", "confidence": 0.88, "treatment": "Apply appropriate fungicide like Propiconazole or Mancozeb if infection is moderate to severe, focusing on upper leaves."},
        {"disease": "Tomato Bacterial Spot", "confidence": 0.92, "treatment": "Use copper-based bactericides. Remove and destroy infected leaves immediately. Avoid overhead watering."},
        {"disease": "Wheat Powdery Mildew", "confidence": 0.85, "treatment": "Apply sulfur-based fungicides or systemic options like Tebuconazole at early signs. Ensure good air circulation."}
    ]
    return random.choice(possible_results)

def forecast_market_price(crop, market_name):
    logger.debug(f"Forecasting market price for {crop} in {market_name} (placeholder).")
    base_prices = {"Wheat": 2100, "Rice": 2800, "Maize": 1900, "Cotton": 6200, "Tomato": 1200, "Default": 2300}
    base_price = base_prices.get(crop, base_prices["Default"])
    current_price = random.uniform(base_price * 0.9, base_price * 1.1)
    forecast_prices = []
    trend_factor = random.uniform(-0.03, 0.03)
    daily_volatility = random.uniform(0.01, 0.06)

    last_price = current_price
    for i in range(7):
        price_change = 1 + (trend_factor * (i+1)/7) + random.uniform(-daily_volatility, daily_volatility)
        next_price = last_price * price_change
        next_price = max(base_price * 0.6, next_price)
        forecast_prices.append(round(next_price, 2))
        last_price = next_price

    trend_suggestion = "Market appears volatile with no clear short-term trend."
    if forecast_prices:
        start_price = forecast_prices[0]
        end_price = forecast_prices[-1]
        if end_price > start_price * 1.04:
            trend_suggestion = "Suggests a potential upward trend in the near term."
        elif end_price < start_price * 0.96:
            trend_suggestion = "Indicates a potential downward trend in the near term."
        elif abs(end_price - start_price) / start_price < 0.015:
            trend_suggestion = "Prices look relatively stable for the next week."

    return {
        "crop": crop,
        "market": market_name,
        "forecast_days": 7,
        "predicted_prices_per_quintal": forecast_prices,
        "trend_suggestion": trend_suggestion
    }


def get_weather_forecast(latitude, longitude, api_key):
    try:
        lat_f = float(latitude)
        lon_f = float(longitude)
    except (ValueError, TypeError):
        logger.warning(f"Invalid latitude ('{latitude}') or longitude ('{longitude}') for weather forecast.")
        return {"status": "error", "message": "Invalid location coordinates provided."}

    if lat_f == 0.0 and lon_f == 0.0:
        logger.info("Weather forecast skipped: Location coordinates are 0.0, 0.0 (likely not set).")
        return {"status": "error", "message": ui_translator("weather_data_error", message="Location not set in profile (or set to 0,0). Cannot fetch weather.")}

    if not api_key:
        logger.warning("Weather API Key not provided for forecast.")
        return {"status": "error", "message": ui_translator("weather_data_error", message="Weather API Key is missing in the configuration.")}

    params = {
        'lat': lat_f,
        'lon': lon_f,
        'appid': api_key,
        'units': 'metric',
        'cnt': 40
    }

    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Weather data fetched successfully for {lat_f:.2f},{lon_f:.2f}.")

        daily_forecasts = defaultdict(lambda: {
            'min_temp': float('inf'),
            'max_temp': float('-inf'),
            'conditions': set(),
            'total_rain': 0.0,
            'alerts': set(),
            'raw_temps': [], 'raw_humidities': [], 'raw_windspeeds': []
        })

        if 'list' not in data or not isinstance(data['list'], list):
            logger.error("Unexpected weather API response format: 'list' key missing or not a list.")
            return {"status": "error", "message": "Unexpected weather API response format."}

        city_info = data.get('city', {})
        location_name = city_info.get('name', f"Lat:{lat_f:.2f},Lon:{lon_f:.2f}")

        for forecast_item in data['list']:
             if not isinstance(forecast_item, dict) or 'dt' not in forecast_item or 'main' not in forecast_item or 'weather' not in forecast_item: continue
             if not isinstance(forecast_item['weather'], list) or not forecast_item['weather']: continue

             main_data = forecast_item.get('main', {})
             weather_data = forecast_item['weather'][0]

             if 'temp_min' not in main_data or 'temp_max' not in main_data or 'description' not in weather_data: continue

             try:
                 dt_object = datetime.datetime.fromtimestamp(forecast_item['dt'])
                 date_str = dt_object.strftime("%Y-%m-%d")
                 temp = float(main_data.get('temp', pd.NA))
                 temp_min = float(main_data['temp_min'])
                 temp_max = float(main_data['temp_max'])
                 humidity = float(main_data.get('humidity', pd.NA))
                 description_formatted = weather_data['description'].capitalize()
                 rain_3h = float(forecast_item.get('rain', {}).get('3h', 0.0))
                 wind_speed = float(forecast_item.get('wind', {}).get('speed', 0.0))

             except (KeyError, ValueError, TypeError) as e:
                 logger.warning(f"Skipping forecast item due to data parsing error ({e}): {forecast_item}")
                 continue

             day_data = daily_forecasts[date_str]
             day_data['min_temp'] = min(day_data['min_temp'], temp_min)
             day_data['max_temp'] = max(day_data['max_temp'], temp_max)
             day_data['conditions'].add(description_formatted)
             day_data['total_rain'] += rain_3h

             if pd.notna(temp): day_data['raw_temps'].append(temp)
             if pd.notna(humidity): day_data['raw_humidities'].append(humidity)
             if pd.notna(wind_speed): day_data['raw_windspeeds'].append(wind_speed)

             if rain_3h > 7: day_data['alerts'].add(f"Heavy rain ({rain_3h:.1f}mm/3hr)")
             elif rain_3h > 2: day_data['alerts'].add(f"Moderate rain ({rain_3h:.1f}mm/3hr)")
             if pd.notna(temp) and temp > 40: day_data['alerts'].add(f"Very High Temp ({temp:.0f}°C)")
             elif pd.notna(temp) and temp > 37: day_data['alerts'].add(f"High Temp ({temp:.0f}°C)")
             elif pd.notna(temp) and temp < 8: day_data['alerts'].add(f"Low Temp ({temp:.0f}°C)")
             if pd.notna(wind_speed) and wind_speed > 17:
                 day_data['alerts'].add(f"Very Strong Wind ({wind_speed * 3.6:.0f} km/h)")
             elif pd.notna(wind_speed) and wind_speed > 12:
                 day_data['alerts'].add(f"Strong Wind ({wind_speed * 3.6:.0f} km/h)")

        processed_summary = []
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)
        sorted_dates = sorted(daily_forecasts.keys())

        days_added = 0
        for date_str in sorted_dates:
            if days_added >= 5: break
            day_data = daily_forecasts[date_str]
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                if date_obj < today: continue
                day_name = date_obj.strftime("%a")
            except ValueError:
                continue

            day_label_key = "day_label_" + day_name.lower()
            day_label_translation = ui_translator(day_label_key, default=day_name)
            if date_obj == today: day_label = ui_translator("label_today", default="Today")
            elif date_obj == tomorrow: day_label = ui_translator("label_tomorrow", default="Tomorrow")
            else: day_label = day_label_translation

            conditions_list = sorted(list(day_data['conditions']))
            if 'Light rain' in conditions_list and 'Rain' in conditions_list: conditions_list.remove('Light rain')
            if 'Few clouds' in conditions_list and ('Scattered clouds' in conditions_list or 'Broken clouds' in conditions_list or 'Overcast clouds' in conditions_list): conditions_list.remove('Few clouds')
            conditions_str = ", ".join(conditions_list) if conditions_list else ui_translator("conditions_unclear")

            rain_str = ""
            if day_data['total_rain'] > 0.1:
                 rain_str = ui_translator("weather_rain_display", value=float(day_data['total_rain']))

            alerts_str = ""
            if day_data['alerts']:
                 alerts_str = ui_translator("weather_alerts_display", alerts_joined=", ".join(sorted(list(day_data['alerts']))))

            min_t_str = f"{day_data['min_temp']:.0f}" if day_data['min_temp'] != float('inf') else ui_translator("value_na")
            max_t_str = f"{day_data['max_temp']:.0f}" if day_data['max_temp'] != float('-inf') else ui_translator("value_na")

            summary_line = (
                f"{day_label} ({date_obj.strftime('%d %b')}): "
                f"Temp {min_t_str}°C / {max_t_str}°C, "
                f"{conditions_str}"
                f"{rain_str}"
                f"{alerts_str}"
            ).strip().replace("  ", " ")
            processed_summary.append(summary_line)
            days_added += 1

        if not processed_summary:
            logger.warning(f"Could not generate daily forecast summary for {lat_f},{lon_f}, though API call succeeded.")
            return {"status": "error", "message": ui_translator("weather_error_summary_generation")}

        return {
            "status": "success",
            "location": location_name,
            "daily_summary": processed_summary
        }

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        error_text = e.response.text if e.response else "No response body"
        logger.error(f"HTTP error fetching weather: {status_code} - {error_text}", exc_info=False)
        if status_code == 401: message_key = "weather_error_401"
        elif status_code == 404: message_key = "weather_error_404"
        elif status_code == 429: message_key = "weather_error_429"
        else: message_key = "weather_error_http"
        message = ui_translator(message_key, status_code=status_code)
        return {"status": "error", "message": ui_translator("weather_data_error", message=message)}

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching weather: {e}", exc_info=True)
        message = ui_translator("weather_error_network")
        return {"status": "error", "message": ui_translator("weather_data_error", message=message)}
    except Exception as e:
        logger.error(f"Unexpected error processing weather data: {e}", exc_info=True)
        message = ui_translator("weather_error_unexpected", error=str(e))
        return {"status": "error", "message": ui_translator("weather_data_error", message=message)}


def generate_final_response_with_history(llm, base_prompt_lines, chat_history_messages, output_language):
    if not llm:
        logger.error("generate_final_response_with_history called without initialized LLM.")
        return ui_translator("llm_init_error")

    system_prompt_content = f"""You are Krishi-Sahayak AI, an expert agricultural advisor specifically for farmers in India. Your goal is to provide insightful, practical, and detailed advice.
Respond ONLY in {output_language}. Do not use any other language.

## Your Task:
Carefully analyze the Farmer's Profile, the provided Context Data (including weather, market prices, etc.), and the Conversation History below.
Synthesize all this information to answer the farmer's MOST RECENT query.

## Key Instructions:
1.  **Integrate Context:** Directly reference relevant details from the farmer's profile (location, soil type, farm size) and the provided data (weather implications, market trends, crop suitability) in your explanation. Don't just repeat the data; interpret its significance for *this* farmer.
2.  **Provide Reasoning:** Explain the 'why' behind your recommendations. If suggesting a crop, explain why it's suitable based on the soil, weather, and possibly market trends. If discussing weather, explain its potential impact (positive or negative) on common agricultural activities or specific crops relevant to the farmer.
3.  **Leverage History:** Refer back to previous turns in the conversation if relevant. Acknowledge earlier advice or questions to provide continuity and build upon the dialogue.
4.  **Actionable & Specific:** Offer clear, concrete steps or options the farmer can take. Avoid vague statements. If multiple options exist, briefly discuss pros and cons. Mention specific product types or practices where appropriate (e.g., "Consider using a nitrogen-rich fertilizer like Urea" instead of just "add fertilizer").
5.  **Tone:** Be knowledgeable, supportive, and practical. Use clear language appropriate for a farmer, but don't oversimplify complex topics. Aim for a detailed and explanatory style.
6.  **Focus:** Address the farmer's *latest* query directly and thoroughly, using the history and context to enrich the answer.

## Farmer Profile & Context for Current Turn:
---
""" + "\n".join(base_prompt_lines) + "\n---\n"

    messages_for_llm = [
        SystemMessage(content=system_prompt_content)
    ]
    messages_for_llm.extend(chat_history_messages)

    logger.debug(f"Generating response using {len(chat_history_messages)} history messages. Output lang: {output_language}")

    try:
        ai_response = llm.invoke(messages_for_llm)

        response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        logger.info("Received response from LLM.")
        return response_content.strip()

    except Exception as e:
        logger.error(f"Exception calling LLM invoke with history: {e}", exc_info=True)
        err_msg = ui_translator("processing_error", e=f"AI communication failure ({type(e).__name__})")
        err_str = str(e).lower()
        if "api key" in err_str or "permission" in err_str or "denied" in err_str or "authenticate" in err_str:
             err_msg = ui_translator("gemini_key_error")
        elif "quota" in err_str or "resource has been exhausted" in err_str:
             err_msg = f"{ui_translator('processing_error', e='API limit reached.')} Please check your quota or try later."
        elif "safety" in err_str or "blocked" in err_str or "finish reason: safety" in err_str:
             reason = "Safety Filter"
             try:
                 if hasattr(e, 'message') and 'prompt feedback' in e.message.lower():
                     parts = e.message.lower().split('block_reason:')
                     if len(parts) > 1:
                         reason_part = parts[1].split(')')[0].split(',')[0].strip()
                         reason = reason_part.capitalize() if reason_part else "Safety Filter"
             except Exception as parse_err:
                  logger.warning(f"Could not parse safety block reason: {parse_err}")
             logger.warning(f"LLM response potentially blocked by API. Reason: {reason}")
             err_msg = f"{ui_translator('processing_error', e=f'Response blocked by content filter ({reason})')}"

        return err_msg


def process_farmer_request(farmer_profile, current_query, chat_history, llm, weather_api_key, output_language):
    static_context_lines = []

    if not farmer_profile or not isinstance(farmer_profile, dict) or not str(farmer_profile.get('name','')).strip():
        logger.error("process_farmer_request called with invalid farmer_profile.")
        return { "status": "error", "farmer_name": ui_translator("unknown_farmer"), "response_text": ui_translator("system_error_label") + ": Internal error - Farmer profile data missing.", "debug_internal_prompt": "" }

    farmer_name = str(farmer_profile['name']).strip()
    query_clean = str(current_query).strip()
    query_lower = query_clean.lower()
    logger.info(f"Processing query for farmer '{farmer_name}': '{query_clean}' | Output Lang: {output_language}")

    lat = farmer_profile.get('latitude', PROFILE_DEFAULT_LAT)
    lon = farmer_profile.get('longitude', PROFILE_DEFAULT_LON)
    soil = farmer_profile.get('soil_type', 'Unknown')
    farm_size = farmer_profile.get('farm_size_ha', 1.0)

    try: lat_f = float(lat); lon_f = float(lon)
    except (ValueError, TypeError): lat_f, lon_f = PROFILE_DEFAULT_LAT, PROFILE_DEFAULT_LON

    if lat_f != 0.0 or lon_f != 0.0:
        location_desc = ui_translator('location_set_description', lat=lat_f, lon=lon_f)
    else:
        location_desc = ui_translator('location_not_set_description')

    size_str = ui_translator("not_set_label")
    if isinstance(farm_size, (int, float)) and pd.notna(farm_size) and farm_size > 0:
        size_str = f"{farm_size:.2f} Ha"

    static_context_lines.append(ui_translator('farmer_context_data', name=farmer_name, location_description=location_desc, soil=soil, size=size_str))
    static_context_lines.append("")

    intent_identified = False
    crop_keywords = ["crop recommend", "suggest crop", "kya ugana", "फसल सुझा", "பயிர்களைப் பரிந்துரை", "ফসল সুপারিশ", "పంటలను సూచిం", "पिके सुचवा", "grow next", "suitable crop", "कौन सी फसल", "எந்தப் பயிர்", "plant next"]
    market_keywords = ["market price", "mandi rate", "bazaar price", "बाजार भाव", "சந்தை விலை", "বাজার দর", "మార్కెట్ ధర", "बाजार भाव", "what price", "selling price", "bhav", "kimat"]
    weather_keywords = ["weather", "forecast", "mausam", "मौसम", "வானிலை", "আবহাওয়া", "వాతావరణం", "हवामान", "rain", "temperature", "barish", "tapman", "humidity", "wind"]
    health_keywords = ["disease", "pest", "infection", "sick plant", "plant health", "रोग", "कीट", "நோய்", "রোগ", "తెగులు", "कीड", "problem with plant", "issue with crop"]

    if any(keyword in query_lower for keyword in weather_keywords):
        intent_identified = True
        logger.info("Intent Detected: Weather Forecast & Implications")
        static_context_lines.append(ui_translator('intent_weather'))
        weather_info = get_weather_forecast(lat_f, lon_f, weather_api_key)
        loc_name_weather = location_desc if weather_info.get('location', None) is None else weather_info.get('location', location_desc)
        static_context_lines.append(ui_translator('context_header_weather', location=loc_name_weather))
        if weather_info.get('status') == 'success':
            summary_list = weather_info.get('daily_summary', [])
            if summary_list:
                static_context_lines.extend([f"- {s}" for s in summary_list])
            else:
                static_context_lines.append(f"- {ui_translator('weather_error_summary_generation')}")
        else:
            error_msg_weather = weather_info.get('message', ui_translator('weather_error_unknown'))
            static_context_lines.append(ui_translator('context_weather_unavailable', error_msg=error_msg_weather))
        static_context_lines.append(ui_translator('context_footer_weather'))
        static_context_lines.append("")

    elif any(keyword in query_lower for keyword in crop_keywords):
        intent_identified = True
        logger.info("Intent Detected: Crop Recommendation")
        static_context_lines.append(ui_translator('intent_crop'))
        region = location_desc
        avg_temp = random.uniform(20, 35)
        avg_rainfall = random.uniform(400, 800)
        season = "Kharif" if 6 <= datetime.datetime.now().month <= 10 else "Rabi"
        suggested_crops = predict_suitable_crops(soil, region, avg_temp, avg_rainfall, season)

        static_context_lines.append(ui_translator('context_header_crop'))
        static_context_lines.append(ui_translator('context_factors_crop', soil=soil, season=season))
        crops_str = ', '.join(suggested_crops) if suggested_crops else ui_translator("no_crops_recommendation")
        static_context_lines.append(ui_translator('context_crop_ideas', crops=crops_str))
        static_context_lines.append(ui_translator('context_footer_crop'))
        static_context_lines.append("")

    elif any(keyword in query_lower for keyword in market_keywords):
        intent_identified = True
        logger.info("Intent Detected: Market Price")
        static_context_lines.append(ui_translator('intent_market'))
        crop = "Wheat"
        if any(c in query_lower for c in ["rice", "chawal", "धान", "चावल", "அரிசி", "চাল", "బియ్యం", "तांदूळ"]): crop = "Rice"
        elif any(c in query_lower for c in ["maize", "makka", "मक्का", "சோளம்", "ভুট্টা", "మొక్కజొన్న", "मका"]): crop = "Maize"
        elif any(c in query_lower for c in ["cotton", "kapas", "कपास", "பருத்தி", "তুলা", "పత్తి", "कापूस"]): crop = "Cotton"
        elif any(c in query_lower for c in ["tomato", "tamatar", "टमाटर", "தக்காளி", "টমেটো", "టమోటా", "टोमॅटो"]): crop = "Tomato"

        market = "Nearby Mandi"
        forecast = forecast_market_price(crop, market)
        prices = forecast.get('predicted_prices_per_quintal', [])
        price_start = float(prices[0]) if prices else 0.0
        price_end = float(prices[-1]) if prices else 0.0

        static_context_lines.append(ui_translator('context_header_market', crop=forecast.get('crop',crop), market=forecast.get('market',market)))
        static_context_lines.append(
            ui_translator(
                'context_data_market',
                days=forecast.get('forecast_days', 0),
                price_start=price_start,
                price_end=price_end,
                trend=forecast.get('trend_suggestion', ui_translator("value_na"))
            )
        )
        static_context_lines.append(ui_translator('context_footer_market'))
        static_context_lines.append("")

    elif any(keyword in query_lower for keyword in health_keywords):
         intent_identified = True
         logger.info("Intent Detected: Plant Health (Placeholder)")
         static_context_lines.append(ui_translator('intent_health'))
         detection = predict_disease_from_image_placeholder()
         conf_f = float(detection.get('confidence', 0.0))

         static_context_lines.append(ui_translator('context_header_health'))
         static_context_lines.append(
             ui_translator(
                 'context_data_health',
                 disease=detection.get('disease', ui_translator("value_na")),
                 confidence=conf_f,
                 treatment=detection.get('treatment', ui_translator("value_na"))
             )
         )
         static_context_lines.append(ui_translator('context_footer_health'))
         static_context_lines.append("")

    if not intent_identified:
        logger.info("Intent Detected: General Question")
        static_context_lines.append(ui_translator('intent_general'))
        static_context_lines.append(ui_translator('context_header_general'))
        static_context_lines.append(ui_translator('context_data_general', query=query_clean))
        static_context_lines.append(ui_translator('context_footer_general'))
        static_context_lines.append("")

    debug_internal_prompt_for_log = "\n".join(static_context_lines)

    if not llm:
        llm_init_err_msg = ui_translator("llm_init_error")
        logger.error(llm_init_err_msg)
        return { "status": "error", "farmer_name": farmer_name, "response_text": llm_init_err_msg, "debug_internal_prompt": debug_internal_prompt_for_log }

    final_response = generate_final_response_with_history(
        llm=llm,
        base_prompt_lines=static_context_lines,
        chat_history_messages=chat_history,
        output_language=output_language
    )

    is_error_response = False
    if final_response is None:
        is_error_response = True
        final_response = ui_translator("processing_error", e="No response received.")
    elif isinstance(final_response, str):
         known_error_keys = ["gemini_key_error", "processing_error", "llm_init_error", "system_error_label", "weather_data_error", "weather_error_", "tts_error_"]
         translated_errors = [ui_translator(k, default=f"_ERR_{k}_") for k in known_error_keys]
         response_lower = final_response.lower()
         if any(err_indicator in response_lower for err_indicator in ["error:", "sorry, i cannot", "warning:", "could not process", "internal error", "invalid api key", "exception:", "blocked by content", "filter", "unable to", "failed to", "api key validation failed"]) or \
            any(translated_err in final_response for translated_err in translated_errors if not translated_err.startswith("_ERR_")):
              is_error_response = True

    if not is_error_response:
        status = "success"
        log_qa(datetime.datetime.now(), farmer_name, output_language, query_clean, final_response, debug_internal_prompt_for_log)
    else:
        status = "error"
        logger.warning(f"Error response generated or LLM failed for farmer '{farmer_name}'. Response/Error: {final_response}")
        error_prefix = f"{ui_translator('system_error_label')}: "
        if not final_response.startswith(error_prefix) and not final_response.lower().startswith("error:"):
            final_response_for_log = error_prefix + final_response
        else:
            final_response_for_log = final_response
        log_qa(datetime.datetime.now(), farmer_name, output_language, query_clean, final_response_for_log, debug_internal_prompt_for_log)

    return {
        "status": status,
        "farmer_name": farmer_name,
        "response_text": final_response,
        "debug_internal_prompt": debug_internal_prompt_for_log
    }


def handle_map_interaction_reference(map_key="folium_map_reference", center=None, zoom=None, allow_click_updates=True):
    st.info(ui_translator("map_instructions"))

    map_center_to_use = center if center else st.session_state.get('map_center', [MAP_DEFAULT_LAT, MAP_DEFAULT_LON])
    map_zoom_to_use = zoom if zoom else st.session_state.get('map_zoom', 5)
    if isinstance(map_center_to_use, tuple): map_center_to_use = list(map_center_to_use)

    m = folium.Map(
        location=map_center_to_use,
        zoom_start=map_zoom_to_use,
        zoom_control=True
        )

    Geocoder(collapsed=False, position='topright', add_marker=False).add_to(m)
    m.add_child(folium.LatLngPopup())

    if allow_click_updates:
        ref_coords = st.session_state.get('map_clicked_ref_coords')
        if ref_coords and ref_coords.get('lat') is not None and ref_coords.get('lon') is not None:
            try:
                ref_lat_f = float(ref_coords['lat'])
                ref_lon_f = float(ref_coords['lon'])
                folium.Marker(
                    [ref_lat_f, ref_lon_f], popup=f"Ref: {ref_lat_f:.6f}, {ref_lon_f:.6f}",
                    tooltip=ui_translator("map_click_reference"), icon=folium.Icon(color='orange', icon='info-sign')
                ).add_to(m)
            except (ValueError, TypeError): logger.warning(f"Invalid reference coords in session: {ref_coords}")

    current_profile = st.session_state.get('current_farmer_profile')
    if current_profile:
        prof_lat = current_profile.get('latitude', PROFILE_DEFAULT_LAT)
        prof_lon = current_profile.get('longitude', PROFILE_DEFAULT_LON)
        if prof_lat != PROFILE_DEFAULT_LAT or prof_lon != PROFILE_DEFAULT_LON:
             try:
                 prof_lat_f = float(prof_lat); prof_lon_f = float(prof_lon)
                 folium.Marker(
                     [prof_lat_f, prof_lon_f], popup=f"Current: {prof_lat_f:.6f}, {prof_lon_f:.6f}",
                     tooltip=ui_translator('active_profile_loc'), icon=folium.Icon(color='blue', icon='home')
                 ).add_to(m)
             except (ValueError, TypeError): logger.warning(f"Invalid profile coords for map marker: lat={prof_lat}, lon={prof_lon}")

    map_data = st_folium(
        m, center=map_center_to_use, zoom=map_zoom_to_use,
        width=700, height=400, key=map_key, returned_objects=[], use_container_width=True
    )

    if map_data:
        new_center_data = map_data.get("center")
        new_zoom = map_data.get("zoom")

        if new_center_data:
            center_coords = None
            if isinstance(new_center_data, dict) and 'lat' in new_center_data and ('lng' in new_center_data or 'lon' in new_center_data):
                 center_coords = [new_center_data['lat'], new_center_data.get('lng', new_center_data.get('lon'))]
            elif isinstance(new_center_data, list) and len(new_center_data) == 2:
                 center_coords = new_center_data

            current_map_center = st.session_state.get('map_center', [0.0, 0.0])
            if center_coords and (abs(center_coords[0] - current_map_center[0]) > 1e-7 or abs(center_coords[1] - current_map_center[1]) > 1e-7):
                  st.session_state.map_center = center_coords

        if new_zoom and new_zoom != st.session_state.get('map_zoom'):
            st.session_state.map_zoom = new_zoom

        last_clicked = map_data.get("last_clicked")
        if allow_click_updates and last_clicked and 'lat' in last_clicked and ('lng' in last_clicked or 'lon' in last_clicked):
            clicked_lat = last_clicked["lat"]
            clicked_lon = last_clicked.get("lng", last_clicked.get("lon"))
            current_ref = st.session_state.get('map_clicked_ref_coords')
            if (not current_ref or current_ref.get('lat') is None or current_ref.get('lon') is None or
                abs(clicked_lat - current_ref.get('lat', 0.0)) > 1e-7 or abs(clicked_lon - current_ref.get('lon', 0.0)) > 1e-7):
                logger.info(f"Map Click (Reference Update via '{map_key}'): Lat={clicked_lat:.6f}, Lon={clicked_lon:.6f}")
                st.session_state.map_clicked_ref_coords = {'lat': clicked_lat, 'lon': clicked_lon}
                st.rerun()

    if allow_click_updates:
        ref_coords_display = st.session_state.get('map_clicked_ref_coords')
        if ref_coords_display and ref_coords_display.get('lat') is not None and ref_coords_display.get('lon') is not None:
             try:
                 ref_lat_f = float(ref_coords_display['lat'])
                 ref_lon_f = float(ref_coords_display['lon'])
                 st.write(f"**{ui_translator('map_click_reference')}** Lat: `{ref_lat_f:.6f}`, Lon: `{ref_lon_f:.6f}`")
             except (ValueError, TypeError):
                 st.caption(f":warning: {ui_translator('map_click_invalid_coords_message')}")
        else:
            st.caption(ui_translator("map_click_prompt_message"))


def display_past_interactions(farmer_name):
    st.header(ui_translator("past_interactions_header", name=farmer_name))
    qa_log_file = QA_LOG_PATH
    if not os.path.exists(qa_log_file):
        st.info(ui_translator("no_past_interactions"))
        return

    try:
        try:
            log_df = pd.read_csv(qa_log_file, encoding='utf-8', keep_default_na=False, low_memory=False)
        except pd.errors.ParserError as parse_err:
             logger.error(f"Parsing error in {qa_log_file}: {parse_err}. Trying recovery.")
             st.warning(f"Warning: Could not parse parts of the QA log file ({parse_err}). Displaying available entries.")
             try:
                 log_df = pd.read_csv(qa_log_file, encoding='utf-8', keep_default_na=False, on_bad_lines='warn')
             except Exception as read_err_fallback:
                  logger.error(f"Fallback reading failed for {qa_log_file}: {read_err_fallback}")
                  st.error(ui_translator("error_displaying_logs", error=f"Could not parse log file: {read_err_fallback}"))
                  return
        except Exception as read_err:
             logger.error(f"Error reading QA log file {qa_log_file}: {read_err}", exc_info=True)
             st.error(ui_translator("error_displaying_logs", error=str(read_err)))
             return

        required_cols = ['timestamp', 'farmer_name', 'language', 'query', 'response']
        missing_cols = [col for col in required_cols if col not in log_df.columns]
        if missing_cols:
             logger.error(f"Past interactions log {qa_log_file} missing columns: {missing_cols}")
             st.error(ui_translator("log_file_corrupt_columns", path=qa_log_file, cols=", ".join(missing_cols)))
             return

        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')

        if log_df.empty:
            st.info(ui_translator("no_past_interactions"))
            return

        farmer_log = log_df[
            log_df['farmer_name'].fillna('').astype(str).str.strip().str.lower() == str(farmer_name).strip().lower()
        ].sort_values(by='timestamp', ascending=False, na_position='last')

        if farmer_log.empty:
            st.info(ui_translator("no_past_interactions"))
            return

        st.markdown("---")

        for _, row in farmer_log.iterrows():
            ts_dt = row.get('timestamp')
            ts = ts_dt.strftime("%Y-%m-%d %H:%M") if pd.notna(ts_dt) else ui_translator("invalid_date_label")
            q = str(row.get('query', ''))
            a = str(row.get('response', ''))
            l = str(row.get('language', ui_translator('value_na')))

            st.markdown(
                ui_translator("log_entry_display", timestamp=ts, query=q, lang=l, response=a),
                unsafe_allow_html=True
            )

    except FileNotFoundError:
         logger.info(f"QA log file {qa_log_file} not found while trying to display interactions.")
         st.info(ui_translator("no_past_interactions"))
    except pd.errors.EmptyDataError:
         logger.info(f"QA log file {qa_log_file} is empty.")
         st.info(ui_translator("no_past_interactions"))
    except Exception as e:
        logger.error(f"Unexpected error reading/displaying past interactions log {qa_log_file} for {farmer_name}: {e}", exc_info=True)
        st.error(ui_translator("error_displaying_logs", error=str(e)))


def get_tts_lang_code(ui_language_name):
    return TTS_LANG_MAP.get(ui_language_name)

def generate_audio_bytes(text_to_speak, lang_code):
    if not GTTS_AVAILABLE:
        logger.error("gTTS library not available, cannot generate audio.")
        return None
    if not text_to_speak or not lang_code:
        logger.warning(f"generate_audio_bytes called with empty text or lang_code.")
        return None

    try:
        tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        logger.info(f"Successfully generated audio bytes in '{lang_code}'.")
        return audio_fp
    except Exception as e:
        logger.error(f"Error generating TTS audio ({lang_code}): {e}", exc_info=True)
        return None


def main():
    if 'selected_language' not in st.session_state: st.session_state.selected_language = "English"
    if 'current_farmer_profile' not in st.session_state: st.session_state.current_farmer_profile = None
    if 'show_new_profile_form' not in st.session_state: st.session_state.show_new_profile_form = False
    if 'map_center' not in st.session_state: st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]
    if 'map_zoom' not in st.session_state: st.session_state.map_zoom = 5
    if 'map_clicked_ref_coords' not in st.session_state: st.session_state.map_clicked_ref_coords = {'lat': None, 'lon': None}
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'form_trigger_name' not in st.session_state: st.session_state.form_trigger_name = None

    if isinstance(st.session_state.map_center, tuple):
        st.session_state.map_center = list(st.session_state.map_center)

    st.set_page_config(
        page_title=ui_translator("page_title"),
        layout="wide",
        initial_sidebar_state="expanded"
    )

    language_options = list(translations.keys())

    def language_change_callback():
        new_lang = st.session_state.widget_lang_select_key
        if st.session_state.selected_language != new_lang:
             st.session_state.selected_language = new_lang
             logger.info(f"Site language MANUALLY changed to {st.session_state.selected_language} via dropdown.")
        else:
            logger.debug("Language change callback triggered, but language is already set.")

    def clear_chat_history():
        st.session_state.chat_history = []
        logger.info("Chat history cleared.")

    with st.sidebar:
        st.header(ui_translator("sidebar_output_header"))
        try:
            current_lang_index = language_options.index(st.session_state.selected_language)
        except ValueError:
             logger.warning(f"Session lang '{st.session_state.selected_language}' not in options, defaulting UI to English.")
             current_lang_index = 0
             if st.session_state.selected_language != "English": st.session_state.selected_language = "English"

        st.selectbox(
            label=ui_translator("select_language_label"), options=language_options,
            key='widget_lang_select_key', index=current_lang_index,
            on_change=language_change_callback
        )
        st.divider()

        st.header(ui_translator("sidebar_config_header"))
        st.text_input(
            ui_translator("gemini_key_label"), type="password",
            value=os.environ.get("GEMINI_API_KEY", st.session_state.get("widget_gemini_key_input", "")),
            help=ui_translator("gemini_key_help"), key="widget_gemini_key_input"
        )
        st.text_input(
            ui_translator("weather_key_label"), type="password",
            value=os.environ.get("WEATHER_API_KEY", st.session_state.get("widget_weather_key_input", "")),
            help=ui_translator("weather_key_help"), key="widget_weather_key_input"
        )
        st.divider()

        st.header(ui_translator("sidebar_profile_header"))
        default_name_val = ""
        if st.session_state.current_farmer_profile and not st.session_state.show_new_profile_form:
            default_name_val = st.session_state.current_farmer_profile.get('name', '')
        elif st.session_state.show_new_profile_form and st.session_state.form_trigger_name:
             default_name_val = st.session_state.form_trigger_name
        elif 'widget_farmer_name_input' in st.session_state:
             default_name_val = st.session_state.widget_farmer_name_input

        st.text_input( ui_translator("farmer_name_label"), key="widget_farmer_name_input", value=default_name_val, placeholder="Type name here..." )

        col1, col2 = st.columns(2)
        load_button_clicked = col1.button(ui_translator("load_profile_button"), key="widget_load_button")
        new_button_clicked = col2.button(ui_translator("new_profile_button"), key="widget_new_button")

        current_entered_name = st.session_state.get("widget_farmer_name_input", "").strip()

        if load_button_clicked or new_button_clicked:
             farmer_db = load_or_create_farmer_db()
             if not current_entered_name:
                 st.warning(ui_translator("name_missing_error"))
             else:
                 profile = find_farmer(farmer_db, current_entered_name)

                 if load_button_clicked:
                     if profile:
                         st.session_state.current_farmer_profile = profile
                         st.session_state.show_new_profile_form = False
                         st.session_state.form_trigger_name = None
                         clear_chat_history()

                         loaded_language = profile.get('language', 'English')
                         language_changed = False
                         if loaded_language in translations and st.session_state.selected_language != loaded_language:
                             st.session_state.selected_language = loaded_language
                             language_changed = True
                             logger.info(f"App language sync to '{loaded_language}' from loaded profile: {profile['name']}.")
                         elif loaded_language not in translations:
                             logger.warning(f"Profile '{profile['name']}' invalid lang '{loaded_language}', keeping app lang {st.session_state.selected_language}.")

                         loaded_lat = profile.get('latitude', PROFILE_DEFAULT_LAT)
                         loaded_lon = profile.get('longitude', PROFILE_DEFAULT_LON)
                         if loaded_lat != 0.0 or loaded_lon != 0.0:
                             st.session_state.map_center = [loaded_lat, loaded_lon]; st.session_state.map_zoom = MAP_CLICK_ZOOM
                         else:
                             st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]; st.session_state.map_zoom = 5
                         st.session_state.map_clicked_ref_coords = {'lat': None, 'lon': None}

                         st.success(ui_translator("profile_loaded_success", name=profile['name']))
                         for key in ['_form_lat_default','_form_lon_default','_form_soil_default','_form_size_default','_form_lang_default']:
                              if key in st.session_state: del st.session_state[key]

                         logger.info(f"Profile loaded for '{profile['name']}'. Rerun (Lang changed: {language_changed}).")
                         st.rerun()
                     else:
                         st.warning(ui_translator("profile_not_found_warning", name=current_entered_name))
                         st.session_state.show_new_profile_form = False

                 elif new_button_clicked:
                     if profile:
                         st.toast(ui_translator("profile_exists_warning", name=current_entered_name), icon="⚠️")
                         st.session_state.current_farmer_profile = profile
                         st.session_state.show_new_profile_form = False
                         st.session_state.form_trigger_name = None
                         clear_chat_history()

                         existing_language = profile.get('language', 'English')
                         language_changed = False
                         if existing_language in translations and st.session_state.selected_language != existing_language:
                             st.session_state.selected_language = existing_language
                             language_changed = True
                             logger.info(f"App language sync to '{existing_language}' from existing profile '{profile['name']}' (via New button).")
                         elif existing_language not in translations:
                              logger.warning(f"Existing profile '{profile['name']}' invalid lang '{existing_language}', keeping app lang {st.session_state.selected_language}.")

                         loaded_lat = profile.get('latitude', PROFILE_DEFAULT_LAT); loaded_lon = profile.get('longitude', PROFILE_DEFAULT_LON)
                         if loaded_lat != 0.0 or loaded_lon != 0.0: st.session_state.map_center = [loaded_lat, loaded_lon]; st.session_state.map_zoom = MAP_CLICK_ZOOM
                         else: st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]; st.session_state.map_zoom = 5
                         st.session_state.map_clicked_ref_coords = {'lat': None, 'lon': None}

                         for key in ['_form_lat_default','_form_lon_default','_form_soil_default','_form_size_default','_form_lang_default']:
                             if key in st.session_state: del st.session_state[key]

                         logger.info(f"Existing profile '{profile['name']}' loaded instead of creating new. Rerun (Lang changed: {language_changed}).")
                         st.rerun()
                     else:
                         st.info(ui_translator("creating_profile_info", name=current_entered_name))
                         st.session_state.show_new_profile_form = True
                         st.session_state.current_farmer_profile = None
                         st.session_state.form_trigger_name = current_entered_name
                         clear_chat_history()

                         ref_coords = st.session_state.get('map_clicked_ref_coords', {})
                         lat_ref = ref_coords.get('lat'); lon_ref = ref_coords.get('lon')
                         st.session_state['_form_lat_default'] = lat_ref if lat_ref is not None else PROFILE_DEFAULT_LAT
                         st.session_state['_form_lon_default'] = lon_ref if lon_ref is not None else PROFILE_DEFAULT_LON
                         st.session_state['_form_soil_default'] = 'Unknown'
                         st.session_state['_form_size_default'] = 1.0
                         st.session_state['_form_lang_default'] = st.session_state.selected_language

                         if lat_ref is not None and lon_ref is not None:
                             st.session_state.map_center = [lat_ref, lon_ref]; st.session_state.map_zoom = MAP_CLICK_ZOOM
                         else:
                             st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]; st.session_state.map_zoom = 5

                         logger.info(f"Showing new profile form for '{current_entered_name}'. Rerun.")
                         st.rerun()
        st.divider()

        form_header_name = st.session_state.get("form_trigger_name")
        if st.session_state.show_new_profile_form and form_header_name:
             st.subheader(ui_translator("new_profile_form_header", name=form_header_name))

             handle_map_interaction_reference(
                 map_key="new_profile_map",
                 center=[st.session_state.get('_form_lat_default', MAP_DEFAULT_LAT), st.session_state.get('_form_lon_default', MAP_DEFAULT_LON)],
                 zoom=st.session_state.map_zoom,
                 allow_click_updates=True
             )

             with st.form("new_profile_details_form", clear_on_submit=False):
                st.markdown(f"**{ui_translator('selected_coords_label')}**")
                default_lat = st.session_state.get('_form_lat_default', PROFILE_DEFAULT_LAT)
                default_lon = st.session_state.get('_form_lon_default', PROFILE_DEFAULT_LON)
                default_lang = st.session_state.get('_form_lang_default', st.session_state.selected_language)
                default_soil = st.session_state.get('_form_soil_default', 'Unknown')
                default_size = st.session_state.get('_form_size_default', 1.0)

                col_lat, col_lon = st.columns(2)
                with col_lat: st.number_input(ui_translator("latitude_label"), min_value=-90.0, max_value=90.0, value=float(default_lat), step=1e-6, format="%.6f", key="form_new_lat")
                with col_lon: st.number_input(ui_translator("longitude_label"), min_value=-180.0, max_value=180.0, value=float(default_lon), step=1e-6, format="%.6f", key="form_new_lon")
                st.markdown("---")

                try: default_lang_index = language_options.index(default_lang)
                except ValueError: default_lang_index = 0
                st.selectbox(ui_translator("pref_lang_label"), options=language_options, index=default_lang_index, key="form_new_lang")

                try: default_soil_index = SOIL_TYPES.index(default_soil)
                except ValueError: default_soil_index = SOIL_TYPES.index('Unknown')
                st.selectbox(ui_translator("soil_type_label"), options=SOIL_TYPES, index=default_soil_index, key="form_new_soil")

                st.number_input(ui_translator("farm_size_label"), value=float(default_size), min_value=0.01, step=0.1, format="%.2f", key="form_new_size")

                submitted_new = st.form_submit_button(ui_translator("save_profile_button"))

                if submitted_new:
                    profile_name_to_save = st.session_state.get("form_trigger_name")
                    if not profile_name_to_save:
                        st.error(ui_translator("system_error_label") + ": Profile name missing during save.")
                        logger.error("New profile form submitted but form_trigger_name was missing.")
                    else:
                        new_profile_data = { 'name': profile_name_to_save, 'language': st.session_state.form_new_lang, 'latitude': st.session_state.form_new_lat, 'longitude': st.session_state.form_new_lon, 'soil_type': st.session_state.form_new_soil, 'farm_size_ha': st.session_state.form_new_size }
                        logger.info(f"Attempting to save new profile for '{profile_name_to_save}'. Data: {new_profile_data}")

                        current_db_state = load_or_create_farmer_db()
                        updated_db = add_or_update_farmer(current_db_state, new_profile_data)

                        if isinstance(updated_db, pd.DataFrame):
                            save_farmer_db(updated_db)
                            saved_profile = find_farmer(updated_db, profile_name_to_save)
                            if saved_profile:
                                st.session_state.current_farmer_profile = saved_profile
                                st.session_state.show_new_profile_form = False
                                st.session_state.form_trigger_name = None
                                clear_chat_history()

                                saved_language = saved_profile.get('language', 'English')
                                lang_changed_on_save = False
                                if saved_language in translations and st.session_state.selected_language != saved_language:
                                     st.session_state.selected_language = saved_language
                                     lang_changed_on_save = True
                                     logger.info(f"App language sync to '{saved_language}' from saved profile: {profile_name_to_save}.")
                                elif saved_language not in translations:
                                     logger.warning(f"Saved profile '{profile_name_to_save}' invalid lang '{saved_language}', keeping app lang {st.session_state.selected_language}.")

                                saved_lat = saved_profile.get('latitude', PROFILE_DEFAULT_LAT); saved_lon = saved_profile.get('longitude', PROFILE_DEFAULT_LON)
                                if saved_lat != 0.0 or saved_lon != 0.0: st.session_state.map_center = [saved_lat, saved_lon]; st.session_state.map_zoom = MAP_CLICK_ZOOM
                                else: st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]; st.session_state.map_zoom = 5
                                st.session_state.map_clicked_ref_coords = {'lat': None, 'lon': None}

                                for key in ['_form_lat_default','_form_lon_default','_form_soil_default','_form_size_default','_form_lang_default']:
                                     if key in st.session_state: del st.session_state[key]

                                st.success(ui_translator("profile_saved_success", name=profile_name_to_save))
                                logger.info(f"New profile saved for '{profile_name_to_save}'. Rerun (Lang changed: {lang_changed_on_save}).")
                                st.rerun()
                            else:
                                logger.error(f"Profile '{profile_name_to_save}' not found immediately after saving.")
                                st.error(ui_translator("profile_reload_error_after_save"))
                                st.session_state.show_new_profile_form = False
                                st.session_state.form_trigger_name = None
                                st.rerun()
                        else:
                            logger.error(f"Failed to get updated DataFrame saving profile '{profile_name_to_save}'.")
                            st.error(ui_translator("db_update_error_on_save"))

        active_profile = st.session_state.current_farmer_profile
        if not st.session_state.show_new_profile_form:
            st.markdown("---")
            if active_profile and isinstance(active_profile, dict):
                st.subheader(ui_translator("active_profile_header"))
                name_disp = active_profile.get('name', ui_translator('value_na'))
                lang_disp = active_profile.get('language', ui_translator('value_na'))
                lat_val = active_profile.get('latitude'); lon_val = active_profile.get('longitude')
                soil_disp = active_profile.get('soil_type', ui_translator('value_na'))
                size_val = active_profile.get('farm_size_ha')

                loc_str = ui_translator('location_not_set_description')
                if pd.notna(lat_val) and pd.notna(lon_val):
                    try:
                        lat_f = float(lat_val); lon_f = float(lon_val)
                        if abs(lat_f) > 1e-9 or abs(lon_f) > 1e-9: loc_str = f"{lat_f:.6f}, {lon_f:.6f}"
                        else: loc_str = ui_translator('location_not_set_description')
                    except (ValueError, TypeError): loc_str = ui_translator('location_not_set_description') + " (Invalid)"

                size_str = ui_translator("not_set_label")
                if pd.notna(size_val):
                    try:
                        size_f = float(size_val)
                        if size_f > 0: size_str = f"{size_f:.2f}"
                    except (ValueError, TypeError): pass

                st.write(f"**{ui_translator('active_profile_name')}:** {name_disp}")
                st.write(f"**{ui_translator('active_profile_lang')}:** {lang_disp}")
                st.write(f"**{ui_translator('active_profile_loc')}:** {loc_str}")
                st.write(f"**{ui_translator('active_profile_soil')}:** {soil_disp}")
                st.write(f"**{ui_translator('active_profile_size')}:** {size_str}")
            elif not st.session_state.show_new_profile_form:
                 st.info(ui_translator("no_profile_loaded_info"))

        # --- Modified Line ---
        st.markdown("<p style='font-size: x-small;'>Made By Advait Kawale</p>", unsafe_allow_html=True)
        # --- End of Modification ---


    st.title(ui_translator("page_title"))
    st.caption(ui_translator("page_caption"))
    st.divider()

    if not st.session_state.current_farmer_profile:
        st.warning(ui_translator("profile_error"))
    else:
        farmer_name = st.session_state.current_farmer_profile.get('name', ui_translator("unknown_farmer"))
        profile_language = st.session_state.current_farmer_profile.get('language', "English")

        tab_chat_label = ui_translator("tab_new_chat")
        tab_history_label = ui_translator("tab_past_interactions")
        tab_edit_label = ui_translator("tab_edit_profile")

        tab1, tab2, tab3 = st.tabs([tab_chat_label, tab_history_label, tab_edit_label])

        with tab1:
            st.header(ui_translator("main_header"))

            for i, message in enumerate(st.session_state.chat_history):
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.markdown(message.content)

                    if role == "assistant" and message.content and not message.content.startswith(f"{ui_translator('system_error_label')}:"):
                        if GTTS_AVAILABLE:
                            tts_lang_code = get_tts_lang_code(profile_language)
                            button_key = f"tts_button_{i}_{role}"

                            if tts_lang_code:
                                if st.button(ui_translator("tts_button_label"), key=button_key, help=ui_translator("tts_button_tooltip", lang=profile_language)):
                                    try:
                                        with st.spinner(ui_translator("tts_generating_spinner", lang=profile_language)):
                                            audio_bytes_io = generate_audio_bytes(message.content, tts_lang_code)
                                        if audio_bytes_io:
                                            st.audio(audio_bytes_io, format="audio/mp3")
                                        else:
                                            st.warning(ui_translator("tts_error_generation", err="Generation failed"))
                                    except Exception as e:
                                        st.error(ui_translator("tts_error_generation", err=str(e)))
                                        logger.error(f"TTS Button Click Error: {e}", exc_info=True)
                            else:
                                st.caption(f"({ui_translator('tts_error_unsupported_lang', lang=profile_language)})")
                        else:
                            st.caption(f"({ui_translator('tts_error_library_missing')})")

            if prompt := st.chat_input(ui_translator("query_label"), key="main_chat_input_widget"):
                logger.info(f"User query: '{prompt}'")
                st.session_state.chat_history.append(HumanMessage(content=prompt))

                gemini_key_present = bool(st.session_state.get("widget_gemini_key_input", "").strip())
                if not gemini_key_present:
                    err_msg_chat = ui_translator("gemini_key_error")
                    st.error(err_msg_chat)
                    st.session_state.chat_history.append(AIMessage(content=f"{ui_translator('system_error_label')}: {err_msg_chat}"))
                    st.rerun()
                else:
                    current_gemini_key = st.session_state.widget_gemini_key_input
                    llm = initialize_llm(current_gemini_key)

                    if llm:
                        output_lang = st.session_state.selected_language
                        current_weather_key = st.session_state.get("widget_weather_key_input","").strip()

                        with st.spinner(ui_translator("thinking_spinner", lang=output_lang)):
                            try:
                                result = process_farmer_request(
                                    farmer_profile=st.session_state.current_farmer_profile,
                                    current_query=prompt,
                                    chat_history=st.session_state.chat_history,
                                    llm=llm,
                                    weather_api_key=current_weather_key,
                                    output_language=output_lang
                                )
                                response_text = result.get('response_text', ui_translator("processing_error", e="Empty response."))
                                logger.info(f"AI Response status: {result.get('status', 'unknown')}. Length: {len(response_text)}")

                                st.session_state.chat_history.append(AIMessage(content=response_text))

                            except Exception as e:
                                logger.exception("Critical error in main chat processing.")
                                error_msg_runtime = ui_translator("processing_error", e=repr(e))
                                st.error(error_msg_runtime)
                                st.session_state.chat_history.append(AIMessage(content=f"{ui_translator('system_error_label')}: {error_msg_runtime}"))

                        st.rerun()
                    else:
                        init_error_msg = ui_translator("llm_init_error")
                        st.session_state.chat_history.append(AIMessage(content=f"{ui_translator('system_error_label')}: {init_error_msg}"))
                        st.rerun()

        with tab2:
            display_past_interactions(farmer_name)

        with tab3:
            st.header(ui_translator("edit_profile_header", name=farmer_name))
            current_profile = st.session_state.get('current_farmer_profile')
            if not current_profile:
                 st.warning(ui_translator("profile_error"))
            else:
                handle_map_interaction_reference(
                     map_key="edit_profile_map",
                     center=[current_profile.get('latitude', MAP_DEFAULT_LAT), current_profile.get('longitude', MAP_DEFAULT_LON)],
                     zoom=MAP_CLICK_ZOOM if (current_profile.get('latitude', 0.0) != 0.0 or current_profile.get('longitude', 0.0) != 0.0) else 5,
                     allow_click_updates=False
                )

                with st.form("edit_profile_form", clear_on_submit=False):
                    st.text_input(ui_translator("profile_name_edit_label"), value=current_profile.get('name', ''), key="edit_form_name_display", disabled=True)

                    st.markdown(f"**{ui_translator('selected_coords_label')}**")
                    col_lat_edit, col_lon_edit = st.columns(2)
                    with col_lat_edit: st.number_input( ui_translator("latitude_label"), min_value=-90.0, max_value=90.0, value=float(current_profile.get('latitude', PROFILE_DEFAULT_LAT)), step=1e-6, format="%.6f", key="edit_form_lat")
                    with col_lon_edit: st.number_input( ui_translator("longitude_label"), min_value=-180.0, max_value=180.0, value=float(current_profile.get('longitude', PROFILE_DEFAULT_LON)), step=1e-6, format="%.6f", key="edit_form_lon")
                    st.markdown("---")

                    current_lang = current_profile.get('language', 'English')
                    try: current_lang_index_edit = language_options.index(current_lang)
                    except ValueError: current_lang_index_edit = 0
                    st.selectbox(ui_translator("pref_lang_label"), options=language_options, index=current_lang_index_edit, key="edit_form_lang")

                    current_soil = current_profile.get('soil_type', 'Unknown')
                    try: current_soil_index_edit = SOIL_TYPES.index(current_soil)
                    except ValueError: current_soil_index_edit = SOIL_TYPES.index('Unknown')
                    st.selectbox(ui_translator("soil_type_label"), options=SOIL_TYPES, index=current_soil_index_edit, key="edit_form_soil")

                    st.number_input( ui_translator("farm_size_label"), value=float(current_profile.get('farm_size_ha', 1.0)), min_value=0.01, step=0.1, format="%.2f", key="edit_form_size")

                    submitted_edit = st.form_submit_button(ui_translator("save_changes_button"))

                    if submitted_edit:
                         profile_name_to_update = current_profile.get('name')
                         if not profile_name_to_update:
                              st.error(ui_translator("system_error_label") + ": Cannot update profile, name is missing.")
                              logger.error("Edit form submitted but current profile name was missing.")
                         else:
                             updated_data = { 'name': profile_name_to_update, 'language': st.session_state.edit_form_lang, 'latitude': st.session_state.edit_form_lat, 'longitude': st.session_state.edit_form_lon, 'soil_type': st.session_state.edit_form_soil, 'farm_size_ha': st.session_state.edit_form_size }
                             logger.info(f"Attempting to update profile for '{profile_name_to_update}'. Data: {updated_data}")

                             current_db_state_edit = load_or_create_farmer_db()
                             updated_db_edit = add_or_update_farmer(current_db_state_edit, updated_data)

                             if isinstance(updated_db_edit, pd.DataFrame):
                                 save_farmer_db(updated_db_edit)
                                 reloaded_profile = find_farmer(updated_db_edit, profile_name_to_update)
                                 if reloaded_profile:
                                     st.session_state.current_farmer_profile = reloaded_profile
                                     st.success(ui_translator("profile_updated_success", name=profile_name_to_update))
                                     logger.info(f"Profile updated successfully for '{profile_name_to_update}'.")

                                     new_language_pref = reloaded_profile.get('language', 'English')
                                     lang_changed_on_edit = False
                                     if new_language_pref != st.session_state.selected_language:
                                         if new_language_pref in translations:
                                             st.session_state.selected_language = new_language_pref
                                             lang_changed_on_edit = True
                                             logger.info(f"App language sync to '{new_language_pref}' after profile edit for {profile_name_to_update}.")
                                         else:
                                              logger.warning(f"Edited profile '{profile_name_to_update}' invalid lang '{new_language_pref}', keeping site lang {st.session_state.selected_language}.")

                                     new_lat = reloaded_profile.get('latitude', PROFILE_DEFAULT_LAT); new_lon = reloaded_profile.get('longitude', PROFILE_DEFAULT_LON)
                                     if new_lat != 0.0 or new_lon != 0.0: st.session_state.map_center = [new_lat, new_lon]; st.session_state.map_zoom = MAP_CLICK_ZOOM
                                     else: st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]; st.session_state.map_zoom = 5

                                     logger.info(f"Rerun after profile edit. Lang changed: {lang_changed_on_edit}")
                                     st.rerun()
                                 else:
                                     logger.error(f"Profile '{profile_name_to_update}' not found immediately after updating.")
                                     st.error(ui_translator("profile_reload_error_after_save") + " (Update)")
                                     st.rerun()
                             else:
                                logger.error(f"Failed to get updated DataFrame when updating profile '{profile_name_to_update}'.")
                                st.error(ui_translator("db_update_error_on_save") + " (Update)")


if __name__ == "__main__":
    logger.info("--- Starting Krishi-Sahayak AI Streamlit App ---")
    data_dir = os.path.dirname(FARMER_CSV_PATH)
    if data_dir and data_dir != "." and not os.path.exists(data_dir):
        try:
             os.makedirs(data_dir)
             logger.info(f"Created data directory: {data_dir}")
        except OSError as e:
             logger.error(f"Could not create data directory {data_dir}: {e}")
    main()