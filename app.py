# -*- coding: utf-8 -*-
"""
streamlit_app.py

Streamlit web application for Krishi-Sahayak AI Demo.
Features:
- Farmer profile management via sorted CSV with fixed columns.
- Dynamic site language selection affecting UI and AI output.
- Uses Langchain & Gemini for tailored AI responses.
- Provides a multi-day weather forecast summary.
- Includes dropdown for soil type in new profile form.
- Shows temporary toast message if trying to create existing profile.
- Logs questions and answers to qa_log.csv.
- Uses recent Q&A history as context for the LLM.
- Location Selection: Browser Geolocation or Interactive Map with distinct Pan/Pick modes.
- FIX: Restructured New Profile section to keep interactive elements outside form.
- FIX: Improved robustness of geolocation parameter handling (page reload inherent).
- FIX: Ensured map picking respects selected mode state.
- FIX: Corrected SyntaxError in weather forecast HTTPError handling.
- FIX: Changed initial map default location to Central India, profile coords default to (0,0).
- FIX: Added flag/UI change after successful geolocation fetch to clarify page refresh.
- FIX: Simplified Lat/Lon persistence in DB save/update functions. Added debug logging.
- FIX: Corrected TypeError in main UI display for location string translation.
"""

# --- CORE IMPORTS ---
import streamlit as st
import os
import datetime
import random
import requests
import pandas as pd
from dotenv import load_dotenv
import logging
from collections import defaultdict
import streamlit.components.v1 as components
from urllib.parse import quote

# --- THIRD-PARTY IMPORTS ---
import folium
try:
    from streamlit_folium import st_folium
except ImportError:
     st.error("Required libraries `folium` and `streamlit-folium` not found. Install: `pip install folium streamlit-folium`")
     st.stop()

# --- Langchain & Gemini Specific Imports ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.error("Required library `langchain-google-genai` not found. Install: `pip install langchain-google-genai pandas streamlit-folium folium python-dotenv requests`")
    LANGCHAIN_AVAILABLE = False
    st.stop()

# --- Constants and Setup ---
load_dotenv()
# Configure logging (set level via environment variable or directly)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Use a specific logger

WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/forecast"
FARMER_CSV_PATH = "farmer_profiles.csv"
QA_LOG_PATH = "qa_log.csv"
CSV_COLUMNS = ['name', 'language', 'latitude', 'longitude', 'soil_type', 'farm_size_ha']
QA_LOG_COLUMNS = ['timestamp', 'farmer_name', 'language', 'query', 'response', 'internal_prompt']

# --- Default Location (Central India for map view, 0,0 for profile data default) ---
MAP_DEFAULT_LAT = 20.5937  # Center of India (approx)
MAP_DEFAULT_LON = 78.9629  # Center of India (approx)
PROFILE_DEFAULT_LAT = 0.0
PROFILE_DEFAULT_LON = 0.0

SOIL_TYPES = [
    "Unknown", "Alluvial Soil", "Black Soil (Regur)", "Red Soil", "Laterite Soil",
    "Desert Soil (Arid Soil)", "Mountain Soil (Forest Soil)", "Saline Soil (Alkaline Soil)",
    "Peaty Soil (Marshy Soil)", "Loamy Soil", "Sandy Loam", "Silt Loam", "Clay Loam",
    "Sandy Clay", "Silty Clay", "Sandy Soil", "Silty Soil", "Clay Soil", "Chalky Soil", "Other"
]

# --- Translations Dictionary (Add keys/translations for ALL languages as needed) ---
translations = {
    "English": {
        "page_title": "Krishi-Sahayak AI", "page_caption": "AI-Powered Agricultural Advice", "sidebar_config_header": "⚙️ Configuration",
        "gemini_key_label": "Google Gemini API Key", "gemini_key_help": "Required for AI responses.", "weather_key_label": "OpenWeatherMap API Key",
        "weather_key_help": "Required for weather forecasts.", "sidebar_profile_header": "👤 Farmer Profile", "farmer_name_label": "Enter Farmer Name",
        "load_profile_button": "Load Profile", "new_profile_button": "New Profile", "profile_loaded_success": "Loaded profile for {name}.",
        "profile_not_found_warning": "No profile found for '{name}'. Click 'New Profile' to create one.", "profile_exists_warning": "Profile for '{name}' already exists. Loading existing profile.",
        "creating_profile_info": "Creating new profile for '{name}'. Fill details below.", "new_profile_form_header": "New Profile for {name}",
        "pref_lang_label": "Preferred Language", "soil_type_label": "Select Soil Type", "location_method_label": "Set Farm Location",
        "loc_method_map": "Pick on Map", "loc_method_current": "Use Current Location (Browser)", "latitude_label": "Latitude", "longitude_label": "Longitude",
        "map_instructions": "Use the controls below the map to switch between modes.", "map_interaction_mode_label": "Map Mode",
        "map_mode_pick": "📍 Pick Location (Click Map)", "map_mode_pan": "🖐️ Pan/Zoom (Move Map)", "selected_coords_label": "Selected Coordinates:",
        "get_current_loc_button": "Get My Current Location",
        "get_current_loc_again_button": "Get Location Again", # New key
        "geolocation_success_message": "✅ Location received.", # New key
        "geolocation_info": "Note: Getting location requires a brief page refresh.", # New key
        "geolocation_error": "Geolocation Error: {error_message}. Please try picking on the map or ensure location permissions are granted.",
        "geolocation_unavailable": "Geolocation is not supported by your browser or is unavailable.", "geolocation_permission_denied": "Permission denied for Geolocation. Please allow location access in your browser settings.",
        "geolocation_timeout": "Could not get location within the time limit.", "farm_size_label": "Farm Size (Hectares)", "save_profile_button": "Save New Profile",
        "profile_saved_success": "Created and loaded profile for {name}.", "name_missing_error": "Farmer name cannot be empty.", "active_profile_header": "✅ Active Profile",
        "active_profile_name": "Name", "active_profile_lang": "Pref. Lang", "active_profile_loc": "Location", "active_profile_soil": "Soil", "active_profile_size": "Size (Ha)",
        "no_profile_loaded_info": "No farmer profile loaded. Enter a name and load or create.", "sidebar_output_header": "🌐 Language Settings", "select_language_label": "Select Site & Response Language",
        "main_header": "Ask your Question", "query_label": "Enter query (e.g., 'Suggest crops', 'Weather?', 'Market price wheat?'):", "get_advice_button": "Get Advice",
        "thinking_spinner": "🤖 Thinking... Asking Krishi-Sahayak AI in {lang}...", "advice_header": "💡 Advice for {name} (in {lang})",
        "profile_error": "❌ Please load or create a farmer profile first using the sidebar.", "query_warning": "⚠️ Please enter a query.", "gemini_key_error": "❌ Please enter your Google Gemini API Key in the sidebar.",
        "processing_error": "A critical error occurred during processing: {e}", "llm_init_error": "Could not initialize the AI model. Check the API key and try again.",
        "debug_expander_title": "Debug Info (Internal Prompt & Data Sent to AI)", "debug_prompt_na": "N/A", "intent_crop": "Farmer Query Intent: Crop Recommendation",
        "intent_market": "Farmer Query Intent: Market Price", "intent_weather": "Farmer Query Intent: Weather Forecast", "intent_health": "Farmer Query Intent: Plant Health Check",
        "intent_general": "Farmer Query Intent: General Question", "crop_suggestion_data": "Crop Suggestion Data: Based on soil '{soil}' in season '{season}', consider: {crops}.",
        "market_price_data": "Market Price Data for {crop} in {market}: Expected price range (per quintal) over next {days} days: {price_start:.2f} to {price_end:.2f}. Trend: {trend}",
        "weather_data_header": "Weather Forecast Data for {location} (Next ~5 days):", "weather_data_error": "Weather Forecast Error: {message}",
        "plant_health_data": "Plant Health Data (Placeholder): Finding: '{disease}' ({confidence:.0%} confidence). Suggestion: {treatment}",
        "general_query_data": "Farmer Query: '{query}'. Provide a concise agricultural answer based on general knowledge.",
        "farmer_context_data": "Farmer Context: Name: {name}, Location: {location_description}, Soil: {soil}.", # Modified key text
        "location_set_description": "Near {lat:.2f},{lon:.2f}", # New key for context when location IS set
        "location_not_set_description": "Location Not Set", # New key for context when location IS NOT set
        "history_header": "Recent Interaction History:", "history_entry": "Past Q ({lang}): {query}\nPast A ({lang}): {response}\n---", "history_not_found": "No recent interaction history found for this farmer.",
    },
     "Hindi": {
        # Ensure ALL keys from English are present, including the new/modified ones
        "get_current_loc_again_button": "फिर से स्थान प्राप्त करें",
        "geolocation_success_message": "✅ स्थान प्राप्त हुआ।",
        "geolocation_info": "ध्यान दें: स्थान प्राप्त करने के लिए पृष्ठ को संक्षिप्त रूप से रीफ्रेश करना आवश्यक है।",
        "loc_method_current": "वर्तमान स्थान का उपयोग करें (ब्राउज़र)", "map_instructions": "मोड बदलने के लिए मानचित्र के नीचे दिए गए नियंत्रणों का उपयोग करें।",
        "map_interaction_mode_label": "मानचित्र मोड", "map_mode_pick": "📍 स्थान चुनें (मानचित्र पर क्लिक करें)", "map_mode_pan": "🖐️ पैन/ज़ूम करें (मानचित्र को स्थानांतरित करें)",
        "selected_coords_label": "चयनित निर्देशांक:", "get_current_loc_button": "मेरा वर्तमान स्थान प्राप्त करें",
        "geolocation_error": "जियोलोकेशन त्रुटि: {error_message}. कृपया मानचित्र पर चुनने का प्रयास करें या सुनिश्चित करें कि स्थान की अनुमति दी गई है।", "geolocation_unavailable": "जियोलोकेशन आपके ब्राउज़र द्वारा समर्थित नहीं है या अनुपलब्ध है।",
        "geolocation_permission_denied": "जियोलोकेशन के लिए अनुमति अस्वीकृत। कृपया अपनी ब्राउज़र सेटिंग्स में स्थान पहुँच की अनुमति दें।", "geolocation_timeout": "समय सीमा के भीतर स्थान प्राप्त नहीं किया जा सका।",
        "page_title": "कृषि-सहायक एआई", "page_caption": "एआई-संचालित कृषि सलाह", "sidebar_config_header": "⚙️ सेटिंग", "gemini_key_label": "गूगल जेमिनी एपीआई कुंजी",
        "gemini_key_help": "एआई प्रतिक्रियाओं के लिए आवश्यक।", "weather_key_label": "ओपनवेदरमैप एपीआई कुंजी", "weather_key_help": "मौसम पूर्वानुमान के लिए आवश्यक।",
        "sidebar_profile_header": "👤 किसान प्रोफाइल", "farmer_name_label": "किसान का नाम दर्ज करें", "load_profile_button": "प्रोफ़ाइल लोड करें", "new_profile_button": "नई प्रोफ़ाइल",
        "profile_loaded_success": "{name} के लिए प्रोफ़ाइल लोड की गई।", "profile_not_found_warning": "'{name}' के लिए कोई प्रोफ़ाइल नहीं मिली। 'नई प्रोफ़ाइल' बनाने के लिए क्लिक करें।",
        "profile_exists_warning": "'{name}' के लिए प्रोफ़ाइल पहले से मौजूद है। मौजूदा प्रोफ़ाइल लोड हो रही है।", "creating_profile_info": "'{name}' के लिए नई प्रोफ़ाइल बनाई जा रही है। नीचे विवरण भरें।",
        "new_profile_form_header": "{name} के लिए नई प्रोफ़ाइल", "pref_lang_label": "पसंदीदा भाषा", "soil_type_label": "मिट्टी का प्रकार चुनें", "location_method_label": "खेत का स्थान निर्धारित करें",
        "loc_method_map": "मानचित्र पर चुनें", "latitude_label": "अक्षांश", "longitude_label": "देशांतर", "farm_size_label": "खेत का आकार (हेक्टेयर)", "save_profile_button": "नई प्रोफ़ाइल सहेजें",
        "profile_saved_success": "{name} के लिए प्रोफ़ाइल बनाई और लोड की गई।", "name_missing_error": "किसान का नाम खाली नहीं हो सकता।", "active_profile_header": "✅ सक्रिय प्रोफ़ाइल",
        "active_profile_name": "नाम", "active_profile_lang": "पसंदीदा भाषा", "active_profile_loc": "स्थान", "active_profile_soil": "मिट्टी", "active_profile_size": "आकार (हेक्टेयर)",
        "no_profile_loaded_info": "कोई किसान प्रोफ़ाइल लोड नहीं हुई। नाम दर्ज करें और लोड करें या बनाएं।", "sidebar_output_header": "🌐 भाषा सेटिंग्स", "select_language_label": "साइट और प्रतिक्रिया भाषा चुनें",
        "main_header": "अपना प्रश्न पूछें", "query_label": "प्रश्न दर्ज करें (जैसे, 'फसल सुझाएं', 'मौसम कैसा है?', 'गेहूं का बाजार भाव?'):", "get_advice_button": "सलाह लें",
        "thinking_spinner": "🤖 सोच रहा हूँ... {lang} में कृषि-सहायक एआई से पूछ रहा हूँ...", "advice_header": "💡 {name} के लिए सलाह ({lang} में)", "profile_error": "❌ कृपया पहले साइडबार का उपयोग करके किसान प्रोफ़ाइल लोड करें या बनाएं।",
        "query_warning": "⚠️ कृपया एक प्रश्न दर्ज करें।", "gemini_key_error": "❌ कृपया साइडबार में अपनी गूगल जेमिनी एपीआई कुंजी दर्ज करें।", "processing_error": "प्रसंस्करण के दौरान एक गंभीर त्रुटि हुई: {e}",
        "llm_init_error": "एआई मॉडल को इनिशियलाइज़ नहीं किया जा सका। एपीआई कुंजी जांचें और पुनः प्रयास करें।", "debug_expander_title": "डीबग जानकारी (एआई को भेजा गया आंतरिक संकेत और डेटा)", "debug_prompt_na": "लागू नहीं",
        "intent_crop": "किसान प्रश्न इरादा: फसल की सिफारिश", "intent_market": "किसान प्रश्न इरादा: बाजार मूल्य", "intent_weather": "किसान प्रश्न इरादा: मौसम पूर्वानुमान", "intent_health": "किसान प्रश्न इरादा: पौधों के स्वास्थ्य की जाँच",
        "intent_general": "किसान प्रश्न इरादा: सामान्य प्रश्न", "crop_suggestion_data": "फसल सुझाव डेटा: '{soil}' मिट्टी और '{season}' मौसम के आधार पर, इन पर विचार करें: {crops}.",
        "market_price_data": "{crop} के लिए {market} में बाजार मूल्य डेटा: अगले {days} दिनों में अपेक्षित मूल्य सीमा (प्रति क्विंटल): {price_start:.2f} से {price_end:.2f} तक। रुझान: {trend}",
        "weather_data_header": "{location} के पास मौसम पूर्वानुमान डेटा (अगले ~5 दिन):", "weather_data_error": "मौसम पूर्वानुमान त्रुटि: {message}",
        "plant_health_data": "पौधों का स्वास्थ्य डेटा (प्लेसहोल्डर): निष्कर्ष: '{disease}' ({confidence:.0%} विश्वास)। सुझाव: {treatment}", "general_query_data": "किसान का प्रश्न: '{query}'. सामान्य ज्ञान के आधार पर संक्षिप्त कृषि उत्तर प्रदान करें।",
        "farmer_context_data": "किसान संदर्भ: नाम: {name}, स्थान: {location_description}, मिट्टी: {soil}.",
        "location_set_description": "{lat:.2f},{lon:.2f} के पास",
        "location_not_set_description": "स्थान निर्धारित नहीं है",
        "history_header": "हाल की बातचीत का इतिहास:", "history_entry": "पिछला प्रश्न ({lang}): {query}\nपिछला उत्तर ({lang}): {response}\n---", "history_not_found": "इस किसान के लिए हाल की बातचीत का कोई इतिहास नहीं मिला।",
    },
    # --- Add COMPLETE translations for Tamil, Bengali, Telugu, Marathi below ---
    "Tamil": { # Placeholder - Needs full translation update
        "page_title": "[Ta]Krishi-Sahayak AI",
        "get_current_loc_again_button": "[Ta]Get Location Again",
        "geolocation_success_message": "[Ta]✅ Location received.",
        "geolocation_info": "[Ta]Note: Getting location requires a brief page refresh.",
        "location_set_description": "[Ta]Near {lat:.2f},{lon:.2f}",
        "location_not_set_description": "[Ta]Location Not Set",
        "farmer_context_data": "[Ta]Farmer Context: Name: {name}, Location: {location_description}, Soil: {soil}.",
        # --- Add all other keys here ---
        "page_caption": "[Ta]AI-Powered Agricultural Advice", "sidebar_config_header": "[Ta]⚙️ Configuration",
        "gemini_key_label": "[Ta]Google Gemini API Key", "gemini_key_help": "[Ta]Required for AI responses.", "weather_key_label": "[Ta]OpenWeatherMap API Key",
        "weather_key_help": "[Ta]Required for weather forecasts.", "sidebar_profile_header": "[Ta]👤 Farmer Profile", "farmer_name_label": "[Ta]Enter Farmer Name",
        "load_profile_button": "[Ta]Load Profile", "new_profile_button": "[Ta]New Profile", "profile_loaded_success": "[Ta]Loaded profile for {name}.",
        "profile_not_found_warning": "[Ta]No profile found for '{name}'. Click 'New Profile' to create one.", "profile_exists_warning": "[Ta]Profile for '{name}' already exists. Loading existing profile.",
        "creating_profile_info": "[Ta]Creating new profile for '{name}'. Fill details below.", "new_profile_form_header": "[Ta]New Profile for {name}",
        "pref_lang_label": "[Ta]Preferred Language", "soil_type_label": "[Ta]Select Soil Type", "location_method_label": "[Ta]Set Farm Location",
        "loc_method_map": "[Ta]Pick on Map", "loc_method_current": "[Ta]Use Current Location (Browser)", "latitude_label": "[Ta]Latitude", "longitude_label": "[Ta]Longitude",
        "map_instructions": "[Ta]Use the controls below the map to switch between modes.", "map_interaction_mode_label": "[Ta]Map Mode",
        "map_mode_pick": "[Ta]📍 Pick Location (Click Map)", "map_mode_pan": "[Ta]🖐️ Pan/Zoom (Move Map)", "selected_coords_label": "[Ta]Selected Coordinates:",
        "get_current_loc_button": "[Ta]Get My Current Location",
        "geolocation_error": "[Ta]Geolocation Error: {error_message}. Please try picking on the map or ensure location permissions are granted.",
        "geolocation_unavailable": "[Ta]Geolocation is not supported by your browser or is unavailable.", "geolocation_permission_denied": "[Ta]Permission denied for Geolocation. Please allow location access in your browser settings.",
        "geolocation_timeout": "[Ta]Could not get location within the time limit.", "farm_size_label": "[Ta]Farm Size (Hectares)", "save_profile_button": "[Ta]Save New Profile",
        "profile_saved_success": "[Ta]Created and loaded profile for {name}.", "name_missing_error": "[Ta]Farmer name cannot be empty.", "active_profile_header": "[Ta]✅ Active Profile",
        "active_profile_name": "[Ta]Name", "active_profile_lang": "[Ta]Pref. Lang", "active_profile_loc": "[Ta]Location", "active_profile_soil": "[Ta]Soil", "active_profile_size": "[Ta]Size (Ha)",
        "no_profile_loaded_info": "[Ta]No farmer profile loaded. Enter a name and load or create.", "sidebar_output_header": "[Ta]🌐 Language Settings", "select_language_label": "[Ta]Select Site & Response Language",
        "main_header": "[Ta]Ask your Question", "query_label": "[Ta]Enter query (e.g., 'Suggest crops', 'Weather?', 'Market price wheat?'):", "get_advice_button": "[Ta]Get Advice",
        "thinking_spinner": "[Ta]🤖 Thinking... Asking Krishi-Sahayak AI in {lang}...", "advice_header": "[Ta]💡 Advice for {name} (in {lang})",
        "profile_error": "[Ta]❌ Please load or create a farmer profile first using the sidebar.", "query_warning": "[Ta]⚠️ Please enter a query.", "gemini_key_error": "[Ta]❌ Please enter your Google Gemini API Key in the sidebar.",
        "processing_error": "[Ta]A critical error occurred during processing: {e}", "llm_init_error": "[Ta]Could not initialize the AI model. Check the API key and try again.",
        "debug_expander_title": "[Ta]Debug Info (Internal Prompt & Data Sent to AI)", "debug_prompt_na": "[Ta]N/A", "intent_crop": "[Ta]Farmer Query Intent: Crop Recommendation",
        "intent_market": "[Ta]Farmer Query Intent: Market Price", "intent_weather": "[Ta]Farmer Query Intent: Weather Forecast", "intent_health": "[Ta]Farmer Query Intent: Plant Health Check",
        "intent_general": "[Ta]Farmer Query Intent: General Question", "crop_suggestion_data": "[Ta]Crop Suggestion Data: Based on soil '{soil}' in season '{season}', consider: {crops}.",
        "market_price_data": "[Ta]Market Price Data for {crop} in {market}: Expected price range (per quintal) over next {days} days: {price_start:.2f} to {price_end:.2f}. Trend: {trend}",
        "weather_data_header": "[Ta]Weather Forecast Data for {location} (Next ~5 days):", "weather_data_error": "[Ta]Weather Forecast Error: {message}",
        "plant_health_data": "[Ta]Plant Health Data (Placeholder): Finding: '{disease}' ({confidence:.0%} confidence). Suggestion: {treatment}",
        "general_query_data": "[Ta]Farmer Query: '{query}'. Provide a concise agricultural answer based on general knowledge.",
        "history_header": "[Ta]Recent Interaction History:", "history_entry": "[Ta]Past Q ({lang}): {query}\nPast A ({lang}): {response}\n---", "history_not_found": "[Ta]No recent interaction history found for this farmer.",
    },
    "Bengali": { # Placeholder - Needs full translation update
        "page_title": "[Bn]Krishi-Sahayak AI",
         "get_current_loc_again_button": "[Bn]Get Location Again",
        "geolocation_success_message": "[Bn]✅ Location received.",
        "geolocation_info": "[Bn]Note: Getting location requires a brief page refresh.",
        "location_set_description": "[Bn]Near {lat:.2f},{lon:.2f}",
        "location_not_set_description": "[Bn]Location Not Set",
        "farmer_context_data": "[Bn]Farmer Context: Name: {name}, Location: {location_description}, Soil: {soil}.",
        # --- Add all other keys here ---
        "page_caption": "[Bn]AI-Powered Agricultural Advice", "sidebar_config_header": "[Bn]⚙️ Configuration",
        "gemini_key_label": "[Bn]Google Gemini API Key", "gemini_key_help": "[Bn]Required for AI responses.", "weather_key_label": "[Bn]OpenWeatherMap API Key",
        "weather_key_help": "[Bn]Required for weather forecasts.", "sidebar_profile_header": "[Bn]👤 Farmer Profile", "farmer_name_label": "[Bn]Enter Farmer Name",
        "load_profile_button": "[Bn]Load Profile", "new_profile_button": "[Bn]New Profile", "profile_loaded_success": "[Bn]Loaded profile for {name}.",
        "profile_not_found_warning": "[Bn]No profile found for '{name}'. Click 'New Profile' to create one.", "profile_exists_warning": "[Bn]Profile for '{name}' already exists. Loading existing profile.",
        "creating_profile_info": "[Bn]Creating new profile for '{name}'. Fill details below.", "new_profile_form_header": "[Bn]New Profile for {name}",
        "pref_lang_label": "[Bn]Preferred Language", "soil_type_label": "[Bn]Select Soil Type", "location_method_label": "[Bn]Set Farm Location",
        "loc_method_map": "[Bn]Pick on Map", "loc_method_current": "[Bn]Use Current Location (Browser)", "latitude_label": "[Bn]Latitude", "longitude_label": "[Bn]Longitude",
        "map_instructions": "[Bn]Use the controls below the map to switch between modes.", "map_interaction_mode_label": "[Bn]Map Mode",
        "map_mode_pick": "[Bn]📍 Pick Location (Click Map)", "map_mode_pan": "[Bn]🖐️ Pan/Zoom (Move Map)", "selected_coords_label": "[Bn]Selected Coordinates:",
        "get_current_loc_button": "[Bn]Get My Current Location",
        "geolocation_error": "[Bn]Geolocation Error: {error_message}. Please try picking on the map or ensure location permissions are granted.",
        "geolocation_unavailable": "[Bn]Geolocation is not supported by your browser or is unavailable.", "geolocation_permission_denied": "[Bn]Permission denied for Geolocation. Please allow location access in your browser settings.",
        "geolocation_timeout": "[Bn]Could not get location within the time limit.", "farm_size_label": "[Bn]Farm Size (Hectares)", "save_profile_button": "[Bn]Save New Profile",
        "profile_saved_success": "[Bn]Created and loaded profile for {name}.", "name_missing_error": "[Bn]Farmer name cannot be empty.", "active_profile_header": "[Bn]✅ Active Profile",
        "active_profile_name": "[Bn]Name", "active_profile_lang": "[Bn]Pref. Lang", "active_profile_loc": "[Bn]Location", "active_profile_soil": "[Bn]Soil", "active_profile_size": "[Bn]Size (Ha)",
        "no_profile_loaded_info": "[Bn]No farmer profile loaded. Enter a name and load or create.", "sidebar_output_header": "[Bn]🌐 Language Settings", "select_language_label": "[Bn]Select Site & Response Language",
        "main_header": "[Bn]Ask your Question", "query_label": "[Bn]Enter query (e.g., 'Suggest crops', 'Weather?', 'Market price wheat?'):", "get_advice_button": "[Bn]Get Advice",
        "thinking_spinner": "[Bn]🤖 Thinking... Asking Krishi-Sahayak AI in {lang}...", "advice_header": "[Bn]💡 Advice for {name} (in {lang})",
        "profile_error": "[Bn]❌ Please load or create a farmer profile first using the sidebar.", "query_warning": "[Bn]⚠️ Please enter a query.", "gemini_key_error": "[Bn]❌ Please enter your Google Gemini API Key in the sidebar.",
        "processing_error": "[Bn]A critical error occurred during processing: {e}", "llm_init_error": "[Bn]Could not initialize the AI model. Check the API key and try again.",
        "debug_expander_title": "[Bn]Debug Info (Internal Prompt & Data Sent to AI)", "debug_prompt_na": "[Bn]N/A", "intent_crop": "[Bn]Farmer Query Intent: Crop Recommendation",
        "intent_market": "[Bn]Farmer Query Intent: Market Price", "intent_weather": "[Bn]Farmer Query Intent: Weather Forecast", "intent_health": "[Bn]Farmer Query Intent: Plant Health Check",
        "intent_general": "[Bn]Farmer Query Intent: General Question", "crop_suggestion_data": "[Bn]Crop Suggestion Data: Based on soil '{soil}' in season '{season}', consider: {crops}.",
        "market_price_data": "[Bn]Market Price Data for {crop} in {market}: Expected price range (per quintal) over next {days} days: {price_start:.2f} to {price_end:.2f}. Trend: {trend}",
        "weather_data_header": "[Bn]Weather Forecast Data for {location} (Next ~5 days):", "weather_data_error": "[Bn]Weather Forecast Error: {message}",
        "plant_health_data": "[Bn]Plant Health Data (Placeholder): Finding: '{disease}' ({confidence:.0%} confidence). Suggestion: {treatment}",
        "general_query_data": "[Bn]Farmer Query: '{query}'. Provide a concise agricultural answer based on general knowledge.",
        "history_header": "[Bn]Recent Interaction History:", "history_entry": "[Bn]Past Q ({lang}): {query}\nPast A ({lang}): {response}\n---", "history_not_found": "[Bn]No recent interaction history found for this farmer.",
    },
    "Telugu": { # Placeholder - Needs full translation update
        "page_title": "[Te]Krishi-Sahayak AI",
        "get_current_loc_again_button": "[Te]Get Location Again",
        "geolocation_success_message": "[Te]✅ Location received.",
        "geolocation_info": "[Te]Note: Getting location requires a brief page refresh.",
        "location_set_description": "[Te]Near {lat:.2f},{lon:.2f}",
        "location_not_set_description": "[Te]Location Not Set",
        "farmer_context_data": "[Te]Farmer Context: Name: {name}, Location: {location_description}, Soil: {soil}.",
        # --- Add all other keys here ---
        "page_caption": "[Te]AI-Powered Agricultural Advice", "sidebar_config_header": "[Te]⚙️ Configuration",
        "gemini_key_label": "[Te]Google Gemini API Key", "gemini_key_help": "[Te]Required for AI responses.", "weather_key_label": "[Te]OpenWeatherMap API Key",
        "weather_key_help": "[Te]Required for weather forecasts.", "sidebar_profile_header": "[Te]👤 Farmer Profile", "farmer_name_label": "[Te]Enter Farmer Name",
        "load_profile_button": "[Te]Load Profile", "new_profile_button": "[Te]New Profile", "profile_loaded_success": "[Te]Loaded profile for {name}.",
        "profile_not_found_warning": "[Te]No profile found for '{name}'. Click 'New Profile' to create one.", "profile_exists_warning": "[Te]Profile for '{name}' already exists. Loading existing profile.",
        "creating_profile_info": "[Te]Creating new profile for '{name}'. Fill details below.", "new_profile_form_header": "[Te]New Profile for {name}",
        "pref_lang_label": "[Te]Preferred Language", "soil_type_label": "[Te]Select Soil Type", "location_method_label": "[Te]Set Farm Location",
        "loc_method_map": "[Te]Pick on Map", "loc_method_current": "[Te]Use Current Location (Browser)", "latitude_label": "[Te]Latitude", "longitude_label": "[Te]Longitude",
        "map_instructions": "[Te]Use the controls below the map to switch between modes.", "map_interaction_mode_label": "[Te]Map Mode",
        "map_mode_pick": "[Te]📍 Pick Location (Click Map)", "map_mode_pan": "[Te]🖐️ Pan/Zoom (Move Map)", "selected_coords_label": "[Te]Selected Coordinates:",
        "get_current_loc_button": "[Te]Get My Current Location",
        "geolocation_error": "[Te]Geolocation Error: {error_message}. Please try picking on the map or ensure location permissions are granted.",
        "geolocation_unavailable": "[Te]Geolocation is not supported by your browser or is unavailable.", "geolocation_permission_denied": "[Te]Permission denied for Geolocation. Please allow location access in your browser settings.",
        "geolocation_timeout": "[Te]Could not get location within the time limit.", "farm_size_label": "[Te]Farm Size (Hectares)", "save_profile_button": "[Te]Save New Profile",
        "profile_saved_success": "[Te]Created and loaded profile for {name}.", "name_missing_error": "[Te]Farmer name cannot be empty.", "active_profile_header": "[Te]✅ Active Profile",
        "active_profile_name": "[Te]Name", "active_profile_lang": "[Te]Pref. Lang", "active_profile_loc": "[Te]Location", "active_profile_soil": "[Te]Soil", "active_profile_size": "[Te]Size (Ha)",
        "no_profile_loaded_info": "[Te]No farmer profile loaded. Enter a name and load or create.", "sidebar_output_header": "[Te]🌐 Language Settings", "select_language_label": "[Te]Select Site & Response Language",
        "main_header": "[Te]Ask your Question", "query_label": "[Te]Enter query (e.g., 'Suggest crops', 'Weather?', 'Market price wheat?'):", "get_advice_button": "[Te]Get Advice",
        "thinking_spinner": "[Te]🤖 Thinking... Asking Krishi-Sahayak AI in {lang}...", "advice_header": "[Te]💡 Advice for {name} (in {lang})",
        "profile_error": "[Te]❌ Please load or create a farmer profile first using the sidebar.", "query_warning": "[Te]⚠️ Please enter a query.", "gemini_key_error": "[Te]❌ Please enter your Google Gemini API Key in the sidebar.",
        "processing_error": "[Te]A critical error occurred during processing: {e}", "llm_init_error": "[Te]Could not initialize the AI model. Check the API key and try again.",
        "debug_expander_title": "[Te]Debug Info (Internal Prompt & Data Sent to AI)", "debug_prompt_na": "[Te]N/A", "intent_crop": "[Te]Farmer Query Intent: Crop Recommendation",
        "intent_market": "[Te]Farmer Query Intent: Market Price", "intent_weather": "[Te]Farmer Query Intent: Weather Forecast", "intent_health": "[Te]Farmer Query Intent: Plant Health Check",
        "intent_general": "[Te]Farmer Query Intent: General Question", "crop_suggestion_data": "[Te]Crop Suggestion Data: Based on soil '{soil}' in season '{season}', consider: {crops}.",
        "market_price_data": "[Te]Market Price Data for {crop} in {market}: Expected price range (per quintal) over next {days} days: {price_start:.2f} to {price_end:.2f}. Trend: {trend}",
        "weather_data_header": "[Te]Weather Forecast Data for {location} (Next ~5 days):", "weather_data_error": "[Te]Weather Forecast Error: {message}",
        "plant_health_data": "[Te]Plant Health Data (Placeholder): Finding: '{disease}' ({confidence:.0%} confidence). Suggestion: {treatment}",
        "general_query_data": "[Te]Farmer Query: '{query}'. Provide a concise agricultural answer based on general knowledge.",
        "history_header": "[Te]Recent Interaction History:", "history_entry": "[Te]Past Q ({lang}): {query}\nPast A ({lang}): {response}\n---", "history_not_found": "[Te]No recent interaction history found for this farmer.",
    },
    "Marathi": { # Placeholder - Needs full translation update
        "page_title": "[Mr]Krishi-Sahayak AI",
         "get_current_loc_again_button": "[Mr]Get Location Again",
        "geolocation_success_message": "[Mr]✅ Location received.",
        "geolocation_info": "[Mr]Note: Getting location requires a brief page refresh.",
        "location_set_description": "[Mr]Near {lat:.2f},{lon:.2f}",
        "location_not_set_description": "[Mr]Location Not Set",
        "farmer_context_data": "[Mr]Farmer Context: Name: {name}, Location: {location_description}, Soil: {soil}.",
        # --- Add all other keys here ---
        "page_caption": "[Mr]AI-Powered Agricultural Advice", "sidebar_config_header": "[Mr]⚙️ Configuration",
        "gemini_key_label": "[Mr]Google Gemini API Key", "gemini_key_help": "[Mr]Required for AI responses.", "weather_key_label": "[Mr]OpenWeatherMap API Key",
        "weather_key_help": "[Mr]Required for weather forecasts.", "sidebar_profile_header": "[Mr]👤 Farmer Profile", "farmer_name_label": "[Mr]Enter Farmer Name",
        "load_profile_button": "[Mr]Load Profile", "new_profile_button": "[Mr]New Profile", "profile_loaded_success": "[Mr]Loaded profile for {name}.",
        "profile_not_found_warning": "[Mr]No profile found for '{name}'. Click 'New Profile' to create one.", "profile_exists_warning": "[Mr]Profile for '{name}' already exists. Loading existing profile.",
        "creating_profile_info": "[Mr]Creating new profile for '{name}'. Fill details below.", "new_profile_form_header": "[Mr]New Profile for {name}",
        "pref_lang_label": "[Mr]Preferred Language", "soil_type_label": "[Mr]Select Soil Type", "location_method_label": "[Mr]Set Farm Location",
        "loc_method_map": "[Mr]Pick on Map", "loc_method_current": "[Mr]Use Current Location (Browser)", "latitude_label": "[Mr]Latitude", "longitude_label": "[Mr]Longitude",
        "map_instructions": "[Mr]Use the controls below the map to switch between modes.", "map_interaction_mode_label": "[Mr]Map Mode",
        "map_mode_pick": "[Mr]📍 Pick Location (Click Map)", "map_mode_pan": "[Mr]🖐️ Pan/Zoom (Move Map)", "selected_coords_label": "[Mr]Selected Coordinates:",
        "get_current_loc_button": "[Mr]Get My Current Location",
        "geolocation_error": "[Mr]Geolocation Error: {error_message}. Please try picking on the map or ensure location permissions are granted.",
        "geolocation_unavailable": "[Mr]Geolocation is not supported by your browser or is unavailable.", "geolocation_permission_denied": "[Mr]Permission denied for Geolocation. Please allow location access in your browser settings.",
        "geolocation_timeout": "[Mr]Could not get location within the time limit.", "farm_size_label": "[Mr]Farm Size (Hectares)", "save_profile_button": "[Mr]Save New Profile",
        "profile_saved_success": "[Mr]Created and loaded profile for {name}.", "name_missing_error": "[Mr]Farmer name cannot be empty.", "active_profile_header": "[Mr]✅ Active Profile",
        "active_profile_name": "[Mr]Name", "active_profile_lang": "[Mr]Pref. Lang", "active_profile_loc": "[Mr]Location", "active_profile_soil": "[Mr]Soil", "active_profile_size": "[Mr]Size (Ha)",
        "no_profile_loaded_info": "[Mr]No farmer profile loaded. Enter a name and load or create.", "sidebar_output_header": "[Mr]🌐 Language Settings", "select_language_label": "[Mr]Select Site & Response Language",
        "main_header": "[Mr]Ask your Question", "query_label": "[Mr]Enter query (e.g., 'Suggest crops', 'Weather?', 'Market price wheat?'):", "get_advice_button": "[Mr]Get Advice",
        "thinking_spinner": "[Mr]🤖 Thinking... Asking Krishi-Sahayak AI in {lang}...", "advice_header": "[Mr]💡 Advice for {name} (in {lang})",
        "profile_error": "[Mr]❌ Please load or create a farmer profile first using the sidebar.", "query_warning": "[Mr]⚠️ Please enter a query.", "gemini_key_error": "[Mr]❌ Please enter your Google Gemini API Key in the sidebar.",
        "processing_error": "[Mr]A critical error occurred during processing: {e}", "llm_init_error": "[Mr]Could not initialize the AI model. Check the API key and try again.",
        "debug_expander_title": "[Mr]Debug Info (Internal Prompt & Data Sent to AI)", "debug_prompt_na": "[Mr]N/A", "intent_crop": "[Mr]Farmer Query Intent: Crop Recommendation",
        "intent_market": "[Mr]Farmer Query Intent: Market Price", "intent_weather": "[Mr]Farmer Query Intent: Weather Forecast", "intent_health": "[Mr]Farmer Query Intent: Plant Health Check",
        "intent_general": "[Mr]Farmer Query Intent: General Question", "crop_suggestion_data": "[Mr]Crop Suggestion Data: Based on soil '{soil}' in season '{season}', consider: {crops}.",
        "market_price_data": "[Mr]Market Price Data for {crop} in {market}: Expected price range (per quintal) over next {days} days: {price_start:.2f} to {price_end:.2f}. Trend: {trend}",
        "weather_data_header": "[Mr]Weather Forecast Data for {location} (Next ~5 days):", "weather_data_error": "[Mr]Weather Forecast Error: {message}",
        "plant_health_data": "[Mr]Plant Health Data (Placeholder): Finding: '{disease}' ({confidence:.0%} confidence). Suggestion: {treatment}",
        "general_query_data": "[Mr]Farmer Query: '{query}'. Provide a concise agricultural answer based on general knowledge.",
        "history_header": "[Mr]Recent Interaction History:", "history_entry": "[Mr]Past Q ({lang}): {query}\nPast A ({lang}): {response}\n---", "history_not_found": "[Mr]No recent interaction history found for this farmer.",
    },
}


# --- Helper Functions ---
# Function `t` (translator)
def t(key, lang_dict, **kwargs):
    default_lang_dict = translations.get("English", {})
    # Fallback to English if the key is missing in the target language
    text_template = lang_dict.get(key, default_lang_dict.get(key, f"[{key}]")) # Return [key] if not in English either
    # Prevent formatting errors for None or NaN
    formatted_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, float) and not pd.notna(v): formatted_kwargs[k] = "N/A" # Or some indicator
        elif v is None: formatted_kwargs[k] = ""
        else: formatted_kwargs[k] = v
    try:
        return text_template.format(**formatted_kwargs)
    except KeyError as e:
        logger.warning(f"Translator 't': Missing format key '{e}' in template for key '{key}'. Template: '{text_template}'")
        return text_template # Return raw template
    except Exception as e:
        logger.error(f"Translator 't': Unexpected format error for key '{key}' with args {formatted_kwargs}: {e}. Template: '{text_template}'", exc_info=False)
        return text_template # Return raw template

# Function load_or_create_farmer_db
def load_or_create_farmer_db():
    if os.path.exists(FARMER_CSV_PATH):
        try:
            df = pd.read_csv(FARMER_CSV_PATH, encoding='utf-8')
            logger.debug(f"Read {len(df)} rows from {FARMER_CSV_PATH}")
            # Ensure all required columns exist, adding if necessary
            for col in CSV_COLUMNS:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' missing in {FARMER_CSV_PATH}, adding with default.")
                    if col == 'latitude': df[col] = PROFILE_DEFAULT_LAT
                    elif col == 'longitude': df[col] = PROFILE_DEFAULT_LON
                    elif col == 'farm_size_ha': df[col] = 1.0
                    elif col == 'soil_type': df[col] = 'Unknown'
                    elif col == 'language': df[col] = 'English'
                    else: df[col] = pd.NA
            # Standardize types and handle NaNs
            df['name'] = df['name'].fillna('').astype(str)
            df = df[df['name'] != ''] # Remove rows with effectively empty names
            df['language'] = df['language'].fillna('English').astype(str)
            df['soil_type'] = df['soil_type'].fillna('Unknown').astype(str)
            # Use PROFILE defaults for lat/lon if conversion fails or value is missing
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(PROFILE_DEFAULT_LAT)
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(PROFILE_DEFAULT_LON)
            df['farm_size_ha'] = pd.to_numeric(df['farm_size_ha'], errors='coerce').fillna(1.0)
            df = df[CSV_COLUMNS] # Ensure correct column order
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

# Function add_or_update_farmer (Simplified Lat/Lon handling)
def add_or_update_farmer(df, profile_data):
    if not isinstance(df, pd.DataFrame):
        logger.error("add_or_update_farmer received non-DataFrame.")
        return pd.DataFrame(columns=CSV_COLUMNS) # Return empty structure

    profile_name_clean = str(profile_data.get('name', '')).strip()
    if not profile_name_clean:
        logger.warning("Attempted to add/update farmer with empty name.")
        return df # Return original df if name is invalid

    name_lower = profile_name_clean.lower()
    if 'name' not in df.columns: df['name'] = '' # Ensure column exists
    existing_indices = df.index[df['name'].fillna('').str.lower() == name_lower].tolist()

    new_data = {}
    # Process incoming profile_data into new_data dict
    for col in CSV_COLUMNS:
        value = profile_data.get(col)

        # Handle Lat/Lon specially - trust the input float more directly
        if col == 'latitude' or col == 'longitude':
            default_val = PROFILE_DEFAULT_LAT if col == 'latitude' else PROFILE_DEFAULT_LON
            if isinstance(value, (int, float)) and pd.notna(value):
                 new_data[col] = float(value) # Use valid float directly
            else:
                 # If not a valid float try conversion, else default
                 num_val = pd.to_numeric(value, errors='coerce')
                 final_val = default_val if pd.isna(num_val) else float(num_val)
                 new_data[col] = final_val
                 if pd.isna(num_val) and value is not None: # Log if we defaulted due to bad input
                      logger.warning(f"Invalid value '{value}' provided for {col} in add_or_update. Using default {default_val}.")

        # Handle Farm Size
        elif col == 'farm_size_ha':
            num_val = pd.to_numeric(value, errors='coerce')
            default_val = 1.0
            value_float = default_val if pd.isna(num_val) else float(num_val)
            # Ensure positive size, fallback to default if zero or negative
            new_data[col] = value_float if value_float > 0 else default_val

        # Handle standard text/categorical fields
        elif col in ['name', 'language', 'soil_type']:
             cleaned_value = str(value).strip() if pd.notna(value) else ''
             # Apply defaults if empty AFTER stripping
             if col == 'name': default = profile_name_clean
             elif col == 'language': default = 'English'
             elif col == 'soil_type': default = 'Unknown'
             else: default = '' # Should not happen
             new_data[col] = cleaned_value if cleaned_value else default
        else: # Should not happen with fixed columns
            new_data[col] = value if value is not None else ''

    logger.debug(f"add_or_update_farmer: Prepared new_data for {profile_name_clean}: {new_data}")

    if not new_data.get('name'):
        logger.error(f"Farmer name became invalid after cleaning for data: {profile_data}")
        return df # Safety check

    # Update or Add Row logic
    if existing_indices:
        idx_to_update = existing_indices[0]
        logger.info(f"Updating profile for '{profile_name_clean}' at index {idx_to_update}")
        try:
            # Ensure all columns exist before assignment
            for col_exist in CSV_COLUMNS:
                if col_exist not in df.columns: df[col_exist] = None # Add if missing
            df.loc[idx_to_update, CSV_COLUMNS] = [new_data[col_assign] for col_assign in CSV_COLUMNS]
        except Exception as e:
            logger.error(f"Error updating DataFrame row at index {idx_to_update}: {e}", exc_info=True)
            st.error(f"Internal error updating profile for {profile_name_clean}")
            return df # Return original df on error
        return df # Return modified df
    else:
        logger.info(f"Adding new profile for '{profile_name_clean}'")
        try:
            new_df_row = pd.DataFrame([new_data], columns=CSV_COLUMNS)
            # Concatenate making sure types don't cause issues if possible
            df_updated = pd.concat([df.astype(new_df_row.dtypes), new_df_row], ignore_index=True)
            return df_updated[CSV_COLUMNS] # Return ordered columns
        except Exception as e:
            logger.error(f"Error concatenating new profile row: {e}", exc_info=True)
            st.error(f"Internal error adding profile for {profile_name_clean}")
            return df # Return original df on error


# Function save_farmer_db (Simplified Lat/Lon handling)
def save_farmer_db(df):
    if not isinstance(df, pd.DataFrame):
        logger.error("Attempted to save a non-DataFrame object as farmer DB.")
        st.error("Internal error: Cannot save profile database.")
        return
    try:
        # Ensure DataFrame contains only the specified columns before saving
        if not all(c in df.columns for c in CSV_COLUMNS):
             logger.warning(f"DataFrame missing required columns before save. Has: {df.columns.tolist()}. Reindexing.")
             # Attempt to fix by reindexing, might introduce NaNs if data was corrupt
             df_to_save = df.reindex(columns=CSV_COLUMNS).copy()
        else:
             df_to_save = df[CSV_COLUMNS].copy()

        # Clean and standardize *non-location* data before saving
        df_to_save['name'] = df_to_save['name'].fillna('').astype(str)
        df_to_save = df_to_save[df_to_save['name'] != ''] # Ensure no empty names saved
        df_to_save['language'] = df_to_save['language'].fillna('English').astype(str)
        df_to_save['soil_type'] = df_to_save['soil_type'].fillna('Unknown').astype(str)
        df_to_save['farm_size_ha'] = pd.to_numeric(df_to_save['farm_size_ha'], errors='coerce').fillna(1.0)

        # Assume Lat/Lon are already correct floats from add_or_update_farmer
        # Ensure numeric type for CSV writing consistency, NaN if error. DO NOT fillna here.
        df_to_save['latitude'] = pd.to_numeric(df_to_save['latitude'], errors='coerce')
        df_to_save['longitude'] = pd.to_numeric(df_to_save['longitude'], errors='coerce')

        logger.debug(f"save_farmer_db: Dataframe state just before sorting and saving:\n{df_to_save.head().to_string()}")

        # Sort by name (case-insensitive) for consistent order
        df_sorted = df_to_save.sort_values(by='name', key=lambda col: col.str.lower(), na_position='last')

        # Save to CSV
        df_sorted.to_csv(FARMER_CSV_PATH, index=False, encoding='utf-8')
        logger.info(f"Successfully saved {len(df_sorted)} profiles to {FARMER_CSV_PATH}.")
    except Exception as e:
        logger.error(f"Error saving farmer profiles to {FARMER_CSV_PATH}: {e}", exc_info=True)
        st.error(f"Could not save farmer profiles: {e}")


# Function find_farmer (Robust version)
def find_farmer(df, name):
    if df is None or df.empty or not isinstance(name, str): return None
    name_clean = name.strip();
    if not name_clean: return None
    name_lower = name_clean.lower()
    if 'name' not in df.columns or not pd.api.types.is_string_dtype(df['name']):
        logger.warning("'name' column missing or not string type in DataFrame during find_farmer.")
        return None
    match = df.loc[df['name'].fillna('').str.lower() == name_lower]
    if not match.empty:
        profile_dict = match.iloc[0].to_dict()
        # Sanitize numeric fields with profile defaults
        for col in ['latitude', 'longitude', 'farm_size_ha']:
            val = profile_dict.get(col)
            default_val = PROFILE_DEFAULT_LAT if col=='latitude' else (PROFILE_DEFAULT_LON if col=='longitude' else 1.0)
            if pd.notna(val):
                num_val = pd.to_numeric(val, errors='coerce')
                profile_dict[col] = default_val if pd.isna(num_val) else float(num_val)
            else: profile_dict[col] = default_val
        # Sanitize string fields
        for col in ['name', 'language', 'soil_type']:
             profile_dict[col] = str(profile_dict.get(col, '')).strip()
             # Apply defaults for critical empty strings
             if col == 'language' and not profile_dict[col]: profile_dict[col] = 'English'
             if col == 'soil_type' and not profile_dict[col]: profile_dict[col] = 'Unknown'
        return profile_dict
    return None

# Function log_qa
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
        log_df_entry.to_csv( QA_LOG_PATH, mode='a', header=not file_exists, index=False, encoding='utf-8')
        logger.info(f"Logged Q&A for farmer '{farmer_name}' to {QA_LOG_PATH}")
    except IOError as e: logger.error(f"IOError logging Q&A to {QA_LOG_PATH}: {e}", exc_info=True);
    except Exception as e: logger.error(f"Unexpected error logging Q&A to {QA_LOG_PATH}: {e}", exc_info=True)

# Function get_recent_qa_history
def get_recent_qa_history(farmer_name, translator_func, language_dict, num_entries=3):
    if not isinstance(farmer_name, str) or not farmer_name.strip(): return ""
    farmer_name_clean = farmer_name.strip().lower()
    if not os.path.exists(QA_LOG_PATH):
        logger.info(f"Q&A log file {QA_LOG_PATH} not found for history retrieval.")
        return "" # No history if log file doesn't exist
    try:
        log_df = pd.read_csv(QA_LOG_PATH, encoding='utf-8', parse_dates=['timestamp'])
        if log_df.empty: return ""
        if not all(col in log_df.columns for col in QA_LOG_COLUMNS):
             logger.warning(f"Q&A log {QA_LOG_PATH} missing required columns. Cannot process history.")
             return ""
        farmer_history = log_df[log_df['farmer_name'].fillna('').str.lower() == farmer_name_clean]
        farmer_history = farmer_history.sort_values(by='timestamp', ascending=False, na_position='last')
        if farmer_history.empty: return translator_func('history_not_found', language_dict)
        recent_history = farmer_history.head(num_entries)
        history_lines = [translator_func('history_header', language_dict)]
        for _, row in recent_history.iterrows():
            query_str = str(row.get('query', '')); response_str = str(row.get('response', '')); lang_str = str(row.get('language', ''))
            query_short = (query_str[:150] + '...') if len(query_str) > 150 else query_str
            response_short = (response_str[:250] + '...') if len(response_str) > 250 else response_str
            history_lines.append( translator_func('history_entry', language_dict, lang=lang_str, query=query_short.replace('\n', ' '), response=response_short.replace('\n', ' ')) )
        formatted_history = "\n".join(history_lines)
        logger.info(f"Retrieved last {len(recent_history)} interactions for farmer '{farmer_name}'.")
        return formatted_history
    except Exception as e:
        logger.error(f"Error reading or processing Q&A log {QA_LOG_PATH}: {e}", exc_info=True)
        return "" # Return empty string on error

# Function initialize_llm
def initialize_llm(api_key):
    if not LANGCHAIN_AVAILABLE:
        st.error("Langchain Google GenAI library not available. Cannot initialize LLM.")
        return None
    if not api_key:
        logger.warning("Attempting to initialize LLM without an API key.")
        st.error("Google Gemini API Key is missing. Please provide it in the sidebar.")
        return None
    try:
        llm = ChatGoogleGenerativeAI( model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        logger.info("Google Gemini LLM object initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"LLM Initialization failed: {e}", exc_info=True)
        error_message = f"Error initializing LLM: {e}"
        if "api_key" in str(e).lower() or "permission" in str(e).lower() or "denied" in str(e).lower() or "authenticate" in str(e).lower(): error_message = f"Error initializing LLM: Invalid API Key or insufficient permissions. Please verify your key. ({e})"
        elif "quota" in str(e).lower(): error_message = f"Error initializing LLM: API quota exceeded. Check your usage limits. ({e})"
        st.error(error_message)
        return None

# --- Placeholder/Example Functions ---
def predict_suitable_crops(soil_type, region, avg_temp, avg_rainfall, season):
    logger.debug(f"Predicting crops: Soil={soil_type}, Region={region}, Temp={avg_temp}, Rain={avg_rainfall}, Season={season}")
    recommendations = []; soil_lower = soil_type.lower() if isinstance(soil_type, str) else ""
    # Simple rule-based suggestions
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
    else: recommendations.extend(["Sorghum", "Local Pulses", "Regional Vegetables", "Fodder Crops"])
    random.shuffle(recommendations); return list(set(recommendations[:3]))

def predict_disease_from_image_placeholder():
    logger.debug("Predicting disease (placeholder function).")
    possible_results = [ {"disease": "Healthy", "confidence": 0.95, "treatment": "No action needed."}, {"disease": "Maize Common Rust", "confidence": 0.88, "treatment": "Apply appropriate fungicide if severe."}, {"disease": "Tomato Bacterial Spot", "confidence": 0.92, "treatment": "Use copper-based bactericides. Remove infected leaves."}]
    return random.choice(possible_results)

def forecast_market_price(crop, market_name):
    logger.debug(f"Forecasting market price for {crop} in {market_name} (placeholder).")
    base_prices = {"Wheat": 2000, "Rice": 2500, "Maize": 1800, "Cotton": 6000, "Tomato": 1500, "Default": 2200}
    base_price = base_prices.get(crop, base_prices["Default"]); current_price = random.uniform(base_price * 0.8, base_price * 1.2); forecast_prices = []; trend = random.choice([-0.02, 0, 0.02]); volatility = random.uniform(0.01, 0.05)
    for _ in range(7): price_change = 1 + trend + random.uniform(-volatility, volatility); current_price *= price_change; current_price = max(base_price * 0.5, current_price); forecast_prices.append(round(current_price, 2))
    if not forecast_prices: trend_suggestion = "Could not determine trend."
    elif forecast_prices[-1] > forecast_prices[0] * 1.05: trend_suggestion = "Prices show a slight upward trend."
    elif forecast_prices[-1] < forecast_prices[0] * 0.95: trend_suggestion = "Prices show a slight downward trend."
    else: trend_suggestion = "Prices look relatively stable."
    return {"crop": crop, "market": market_name, "forecast_days": 7, "predicted_prices_per_quintal": forecast_prices, "trend_suggestion": trend_suggestion}

# Function get_weather_forecast (Checks for 0,0 location)
def get_weather_forecast(latitude, longitude, api_key):
    if latitude == 0.0 and longitude == 0.0:
        logger.info("Weather forecast skipped: Location not set (0,0).")
        return {"status": "error", "message": "Location not set in profile. Cannot fetch weather."}
    if not api_key:
        logger.warning("Weather API Key not provided for forecast.")
        return {"status": "error", "message": "Weather API Key is missing in the configuration."}

    params = {'lat': latitude, 'lon': longitude, 'appid': api_key, 'units': 'metric', 'cnt': 40}
    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=15)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        logger.info(f"Weather data fetched successfully for {latitude},{longitude}.")

        daily_forecasts = defaultdict(lambda: {'min_temp': float('inf'), 'max_temp': float('-inf'), 'conditions': set(), 'total_rain': 0.0, 'alerts': set()})
        if 'list' not in data or not isinstance(data['list'], list):
            logger.error("Unexpected weather API response format: 'list' key missing or not a list.")
            return {"status": "error", "message": "Unexpected weather API response format."}

        city_info = data.get('city', {})
        location_name = city_info.get('name', f"Lat:{latitude:.2f},Lon:{longitude:.2f}")

        for forecast_item in data['list']:
             if not isinstance(forecast_item, dict) or 'dt' not in forecast_item or 'main' not in forecast_item or 'weather' not in forecast_item: continue
             if not isinstance(forecast_item['weather'], list) or not forecast_item['weather']: continue
             main_data = forecast_item.get('main', {}); weather_data = forecast_item['weather'][0]
             if 'temp' not in main_data or 'temp_min' not in main_data or 'temp_max' not in main_data or 'description' not in weather_data: continue
             try:
                 dt_object = datetime.datetime.fromtimestamp(forecast_item['dt']); date_str = dt_object.strftime("%Y-%m-%d")
                 temp = main_data['temp']; temp_min = main_data['temp_min']; temp_max = main_data['temp_max']
                 description_formatted = weather_data['description'].capitalize()
                 rain_3h = forecast_item.get('rain', {}).get('3h', 0.0); wind_speed = forecast_item.get('wind', {}).get('speed', 0.0)
             except (KeyError, ValueError, TypeError) as e: logger.warning(f"Skipping item due to parsing error: {e}"); continue # Skip item

             day_data = daily_forecasts[date_str]
             day_data['min_temp'] = min(day_data['min_temp'], float(temp_min))
             day_data['max_temp'] = max(day_data['max_temp'], float(temp_max))
             day_data['conditions'].add(description_formatted); day_data['total_rain'] += float(rain_3h)
             # Basic alerts
             if float(rain_3h) > 5: day_data['alerts'].add(f"Heavy rain ({rain_3h:.1f}mm/3hr)")
             if float(temp) > 38: day_data['alerts'].add(f"High temp ({temp:.0f}°C)")
             if float(temp) < 5: day_data['alerts'].add(f"Low temp ({temp:.0f}°C)")
             if float(wind_speed) > 15: day_data['alerts'].add(f"Strong wind ({wind_speed:.1f} m/s)")

        processed_summary = []
        today = datetime.date.today().strftime("%Y-%m-%d"); tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        sorted_dates = sorted(daily_forecasts.keys())
        for date_str in sorted_dates[:5]: # Limit days
            day_data = daily_forecasts[date_str]
            try: day_name = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
            except ValueError: continue
            if date_str == today: day_label = "Today"
            elif date_str == tomorrow: day_label = "Tomorrow"
            else: day_label = day_name
            conditions_list = sorted(list(day_data['conditions'])); conditions_str = ", ".join(conditions_list) if conditions_list else "Conditions unclear"
            rain_str = f", Rain: {day_data['total_rain']:.1f}mm" if day_data['total_rain'] > 0.1 else ""
            alerts_str = ". Alerts: " + ", ".join(sorted(list(day_data['alerts']))) if day_data['alerts'] else ""
            min_t_str = f"{day_data['min_temp']:.0f}" if day_data['min_temp'] != float('inf') else "N/A"
            max_t_str = f"{day_data['max_temp']:.0f}" if day_data['max_temp'] != float('-inf') else "N/A"
            summary_line = (f"{day_label} ({day_name}): Temp {min_t_str}°C / {max_t_str}°C, {conditions_str}{rain_str}{alerts_str}")
            processed_summary.append(summary_line)
        if not processed_summary: return {"status": "error", "message": "Could not generate daily forecast summary."}
        return {"status": "success", "location": location_name, "daily_summary": processed_summary}
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code; error_text = e.response.text
        logger.error(f"HTTP error fetching weather: {status_code} - {error_text}", exc_info=False)
        message = f"Weather service error (Code: {status_code}). Check API key or service status."
        if status_code == 401: message = "Weather API Key invalid or unauthorized. Please check the key in the sidebar."
        elif status_code == 404: message = f"Weather data not found for location ({latitude},{longitude})."
        elif status_code == 429: message = "Weather API rate limit exceeded. Please try again later."
        return {"status": "error", "message": message}
    except requests.exceptions.RequestException as e: logger.error(f"Network error fetching weather: {e}", exc_info=True); return {"status": "error", "message": "Network error connecting to weather service."}
    except Exception as e: logger.error(f"Unexpected error processing weather data: {e}", exc_info=True); return {"status": "error", "message": f"An unexpected error occurred while processing weather data: {e}"}

# Function generate_final_response
def generate_final_response(llm, internal_prompt, output_language):
    if not llm:
        logger.error("generate_final_response called without initialized LLM.")
        return f"Error: AI Model is not initialized properly (in {output_language})."

    # Updated prompt to use location_description
    full_prompt = f"""You are Krishi-Sahayak AI, a helpful and practical farming advisor for farmers in India. Your goal is to provide clear, concise, and actionable advice based on the context provided below, which addresses the farmer's likely need or query. Respond ONLY in {output_language}. Pay attention if the location context says 'Location Not Set'.

Context and Data:
---
{internal_prompt}
---

Task: Based *only* on the context above, provide a helpful and actionable response relevant to farming in {output_language}.
- If weather data is present (meaning location was set), summarize the key conditions (temperature trends, rain chances, alerts) for the next few days and suggest relevant farming actions (e.g., irrigation planning, taking precautions).
- If weather context indicates an error or that location was not set, clearly state that specific weather advice cannot be given and why.
- If crop suggestions are given, present them clearly.
- If market data is provided, explain the price trend simply.
- If plant health info is present, state the finding and suggestion.
- If it's a general query, provide a concise agricultural answer using the context. If location isn't set, make advice more general.
- **If 'Recent Interaction History' is provided, consider it to maintain consistency. Avoid repeating the exact same advice unless conditions haven't changed or the query asks again. Do not list the history back.**
- Synthesize information; do NOT just repeat the raw data.
- Do NOT suggest doing external research. Provide the answer based only on the given context.
- Ensure the entire response is in {output_language}."""

    logger.info(f"Generating final response in {output_language}.")
    try:
        ai_response = llm.invoke(full_prompt)
        response_content = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        logger.info("Received response from LLM.")
        return response_content.strip()
    except Exception as e:
        logger.error(f"Exception calling LLM invoke: {e}", exc_info=True)
        err_msg = f"Sorry, there was a communication error with the AI assistant in {output_language}. (Error Type: {type(e).__name__})"
        err_str = str(e).lower()
        if "api key" in err_str or "permission" in err_str or "denied" in err_str or "authenticate" in err_str: err_msg = f"Error: Could not authenticate with the AI service due to an invalid API key or permission issues (in {output_language}). Please check the key."
        elif "quota" in err_str or "resource has been exhausted" in err_str: err_msg = f"Error: The AI service API quota has been exceeded or the rate limit reached (in {output_language}). Please try again later or check your plan."
        elif hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback') and hasattr(e.response.prompt_feedback, 'block_reason') and e.response.prompt_feedback.block_reason: err_msg = f"Warning: The AI's response was blocked due to content safety settings ({e.response.prompt_feedback.block_reason}) in {output_language}."
        elif "candidate" in err_str and "finish reason" in err_str and ("safety" in err_str or "block" in err_str): err_msg = f"Warning: The AI's response might have been blocked by safety filters in {output_language}."
        st.error(err_msg)
        return err_msg

# Function process_farmer_request
def process_farmer_request(farmer_profile, text_query, llm, weather_api_key, output_language, translator_func):
    internal_prompt_lines = []
    if not farmer_profile or not isinstance(farmer_profile, dict) or not str(farmer_profile.get('name','')).strip():
        logger.error("process_farmer_request called with invalid farmer_profile.")
        return {"status": "error", "response_text": "Internal error: Farmer profile data is missing or invalid.", "debug_internal_prompt": ""}

    farmer_name = str(farmer_profile['name']).strip()
    query_clean = str(text_query).strip(); query_lower = query_clean.lower()
    logger.info(f"Processing query for farmer '{farmer_name}': '{query_clean}' | Output Language: {output_language}")

    current_lang_dict = translations.get(output_language, translations["English"])
    lat = farmer_profile.get('latitude', PROFILE_DEFAULT_LAT)
    lon = farmer_profile.get('longitude', PROFILE_DEFAULT_LON)
    soil = farmer_profile.get('soil_type', 'Unknown')
    try: lat_f = float(lat); lon_f = float(lon)
    except (ValueError, TypeError): lat_f, lon_f = PROFILE_DEFAULT_LAT, PROFILE_DEFAULT_LON

    # --- Build Context ---
    # 1. Farmer Context (use descriptive location based on coords)
    if lat_f != 0.0 or lon_f != 0.0:
        location_desc = translator_func('location_set_description', current_lang_dict, lat=lat_f, lon=lon_f)
    else:
        location_desc = translator_func('location_not_set_description', current_lang_dict)
    internal_prompt_lines.append(translator_func('farmer_context_data', current_lang_dict, name=farmer_name, location_description=location_desc, soil=soil))

    # 2. History
    history_context = get_recent_qa_history(farmer_name, translator_func, current_lang_dict, num_entries=3)
    if history_context and history_context != translator_func('history_not_found', current_lang_dict):
        internal_prompt_lines.append("\n" + history_context + "\n")
    #else: internal_prompt_lines.append("\n" + translator_func('history_not_found', current_lang_dict) + "\n") # Optional: Add 'not found' explicitly

    # 3. Current Query
    internal_prompt_lines.append(f"--- Current Interaction ---")
    internal_prompt_lines.append(f"Current Farmer Query ({output_language}): {query_clean}\n")

    # 4. Data based on Intent
    intent_identified = False
    # Crop Rec Intent
    if any(keyword in query_lower for keyword in ["crop recommend", "suggest crop", "kya ugana", "फसल सुझाएं", "பயிர்களைப் பரிந்துரை", "ফসল সুপারিশ করুন", "పంటలను సూచించండి", "पिके सुचवा"]):
        intent_identified = True; logger.info("Intent Detected: Crop Recommendation")
        internal_prompt_lines.append(translator_func('intent_crop', current_lang_dict))
        region = location_desc # Use descriptive location
        avg_temp = random.uniform(20, 35); avg_rainfall = random.uniform(400, 800); season = "Kharif" if 6 <= datetime.datetime.now().month <= 10 else "Rabi"
        crops = predict_suitable_crops(soil, region, avg_temp, avg_rainfall, season)
        internal_prompt_lines.append(translator_func('crop_suggestion_data', current_lang_dict, soil=soil, season=season, crops=', '.join(crops) if crops else "None specific"))
    # Market Price Intent
    elif any(keyword in query_lower for keyword in ["market price", "bhav kya hai", "rate kya hai", "बाजार भाव", "சந்தை விலை", "বাজারদর", "మార్కెట్ ధర", "बाजारभाव"]):
        intent_identified = True; logger.info("Intent Detected: Market Price")
        internal_prompt_lines.append(translator_func('intent_market', current_lang_dict))
        crop = "Wheat"; market = "Local Market"; words = query_lower.split() # Simple crop extraction
        if any(c in words for c in ["rice", "chawal", "चावल", "அரிசி", "চাল", "బియ్యం", "तांदूळ"]): crop = "Rice"
        elif any(c in words for c in ["maize", "makka", "मक्का", "சோளம்", "ভুট্টা", "మొక్కజొన్న", "मका"]): crop = "Maize"
        elif any(c in words for c in ["cotton", "kapas", "कपास", "பருத்தி", "তুলা", "పత్తి", "कापूस"]): crop = "Cotton"
        elif any(c in words for c in ["tomato", "tamatar", "टमाटर", "தக்காளி", "টমেটো", "టమోటా", "टोमॅटो"]): crop = "Tomato"
        forecast = forecast_market_price(crop, market); prices = forecast.get('predicted_prices_per_quintal', [0.0, 0.0]); price_start = float(prices[0]) if prices else 0.0; price_end = float(prices[-1]) if prices else 0.0
        internal_prompt_lines.append(translator_func('market_price_data', current_lang_dict, crop=forecast.get('crop','N/A'), market=forecast.get('market','N/A'), days=forecast.get('forecast_days',0), price_start=price_start, price_end=price_end, trend=forecast.get('trend_suggestion','N/A')))
    # Weather Intent
    elif any(keyword in query_lower for keyword in ["weather", "mausam", "मौसम", "வானிலை", "আবহাওয়া", "వాతావరణం", "हवामान"]):
        intent_identified = True; logger.info("Intent Detected: Weather Forecast")
        internal_prompt_lines.append(translator_func('intent_weather', current_lang_dict))
        weather_info = get_weather_forecast(lat_f, lon_f, weather_api_key) # Function now handles 0,0 case
        if weather_info.get('status') == 'success':
            internal_prompt_lines.append(translator_func('weather_data_header', current_lang_dict, location=weather_info.get('location', 'your area')))
            internal_prompt_lines.extend(weather_info.get('daily_summary', [translator_func('debug_prompt_na', current_lang_dict)]))
        else: internal_prompt_lines.append(translator_func('weather_data_error', current_lang_dict, message=weather_info.get('message', 'Unknown weather error')))
    # Plant Health Intent
    elif any(keyword in query_lower for keyword in ["disease", "pest", "problem", "rog", "bimari", "कीट", "நோய்", "রোগ", "తెగులు", "कीड", "समस्या"]):
         intent_identified = True; logger.info("Intent Detected: Plant Health (Placeholder)")
         internal_prompt_lines.append(translator_func('intent_health', current_lang_dict))
         detection = predict_disease_from_image_placeholder(); conf = detection.get('confidence',0.0); conf_f = float(conf) if isinstance(conf, (int, float)) else 0.0
         internal_prompt_lines.append(translator_func('plant_health_data', current_lang_dict, disease=detection.get('disease','N/A'), confidence=conf_f, treatment=detection.get('treatment','N/A')))
    # General Intent (Fallback)
    if not intent_identified:
        logger.info("Intent Detected: General Question")
        internal_prompt_lines.append(translator_func('intent_general', current_lang_dict))
        internal_prompt_lines.append(translator_func('general_query_data', current_lang_dict, query=query_clean))

    # --- Combine and Generate Response ---
    internal_prompt = "\n".join(internal_prompt_lines)
    logger.debug(f"Internal prompt being sent to LLM:\n---\n{internal_prompt}\n---")

    if not llm:
        llm_init_err_msg = translator_func("llm_init_error", current_lang_dict)
        logger.error(llm_init_err_msg)
        return {"status": "error", "farmer_name": farmer_name, "response_text": llm_init_err_msg, "debug_internal_prompt": internal_prompt}

    final_response = generate_final_response(llm, internal_prompt, output_language)
    is_error_response = (final_response is None or any(final_response.startswith(prefix) for prefix in ["Error:", "Sorry,", "Warning:", "Could not", "Internal error"]))

    if final_response and not is_error_response:
        log_qa(datetime.datetime.now(), farmer_name, output_language, query_clean, final_response, internal_prompt)
        status = "success"
    elif not final_response:
        logger.warning(f"LLM generated empty response for farmer '{farmer_name}'. Query: {query_clean}")
        final_response = translator_func("processing_error", current_lang_dict, e="AI assistant did not provide a response.")
        status = "error"
    else:
        logger.warning(f"Error response generated or LLM failed for farmer '{farmer_name}'. Error: {final_response}")
        status = "error" # Already an error message

    return {
        "status": status,
        "farmer_name": farmer_name,
        "response_text": final_response if final_response else translator_func("processing_error", current_lang_dict, e="No response generated."),
        "debug_internal_prompt": internal_prompt
    }


# --- Streamlit UI ---
def main():
    # --- Initialize Session State ---
    st.session_state.setdefault('selected_language', "English")
    st.session_state.setdefault('current_farmer_profile', None)
    st.session_state.setdefault('show_new_profile_form', False)
    st.session_state.setdefault('new_profile_location_method', "map")
    st.session_state.setdefault('map_interaction_mode', "pick")
    st.session_state.setdefault('new_profile_coords', {'lat': PROFILE_DEFAULT_LAT, 'lon': PROFILE_DEFAULT_LON})
    st.session_state.setdefault('map_center', [MAP_DEFAULT_LAT, MAP_DEFAULT_LON])
    st.session_state.setdefault('map_zoom', 5)
    st.session_state.setdefault('params_processed', False)
    st.session_state.setdefault('geolocation_fetched_successfully', False) # Flag for UI change

    # --- Process Geolocation Query Params ---
    query_params = st.query_params
    if not st.session_state.params_processed and "lat" in query_params and "lon" in query_params:
        try:
            geo_lat = float(query_params.get("lat"))
            geo_lon = float(query_params.get("lon"))
            logger.info(f"Received Geolocation via URL params: lat={geo_lat}, lon={geo_lon}")
            st.session_state.new_profile_coords = {'lat': geo_lat, 'lon': geo_lon}
            st.session_state.map_center = [geo_lat, geo_lon] # Center map
            st.session_state.map_zoom = 14 # Zoom in
            st.session_state.new_profile_location_method = "current" # Switch mode
            if not st.session_state.show_new_profile_form: st.session_state.show_new_profile_form = True
            st.session_state.geolocation_fetched_successfully = True # SET SUCCESS FLAG!
            st.session_state.params_processed = True
            st.query_params.clear() # Clear params after processing
            st.rerun() # Rerun to update UI
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Error processing geolocation query params (lat='{query_params.get('lat')}', lon='{query_params.get('lon')}'): {e}")
            temp_lang_dict = translations.get(st.session_state.selected_language, translations["English"])
            st.warning(t("geolocation_error",temp_lang_dict, error_message=f"Invalid coordinate data in URL ({e})"))
            st.session_state.params_processed = True; st.session_state.geolocation_fetched_successfully = False
            st.query_params.clear()

    # --- Language Setup ---
    language_options = list(translations.keys())
    if st.session_state.selected_language not in language_options: st.session_state.selected_language = "English"
    current_lang_dict = translations.get(st.session_state.selected_language, translations["English"])
    # IMPORTANT: This lambda captures the current_lang_dict from the scope it's defined in.
    # It should be called with only the key and optional **kwargs.
    ui_translator = lambda k, **kwargs: t(k, current_lang_dict, **kwargs)
    st.set_page_config(page_title=ui_translator("page_title"), layout="wide")
    def language_change_callback():
        st.session_state.selected_language = st.session_state.widget_lang_select_key
        logger.info(f"Site language state updated to {st.session_state.selected_language} via callback.")
        st.session_state.params_processed = False

    # ================== SIDEBAR ==================
    with st.sidebar:
        # Language, API Keys, Profile Name Input
        st.header(ui_translator("sidebar_output_header"))
        try: lang_index = language_options.index(st.session_state.selected_language)
        except ValueError: lang_index = 0
        st.selectbox(label=ui_translator("select_language_label"), options=language_options, key='widget_lang_select_key', index=lang_index, on_change=language_change_callback)
        st.header(ui_translator("sidebar_config_header"))
        st.text_input(ui_translator("gemini_key_label"), type="password", value=os.environ.get("GEMINI_API_KEY", st.session_state.get("widget_gemini_key","")), help=ui_translator("gemini_key_help"), key="widget_gemini_key")
        st.text_input(ui_translator("weather_key_label"), type="password", value=os.environ.get("WEATHER_API_KEY", st.session_state.get("widget_weather_key", "")), help=ui_translator("weather_key_help"), key="widget_weather_key")
        st.divider()
        st.header(ui_translator("sidebar_profile_header"))
        default_name = st.session_state.get('widget_farmer_name_input', (st.session_state.current_farmer_profile['name'] if st.session_state.current_farmer_profile else ""))
        st.text_input(ui_translator("farmer_name_label"), key="widget_farmer_name_input", value=default_name)
        col1, col2 = st.columns(2)
        load_button_clicked = col1.button(ui_translator("load_profile_button"), key="widget_load_button")
        new_button_clicked = col2.button(ui_translator("new_profile_button"), key="widget_new_button")
        farmer_db = load_or_create_farmer_db()
        current_entered_name = st.session_state.widget_farmer_name_input.strip()

        # Profile Loading / New Profile Actions
        if load_button_clicked:
            if current_entered_name:
                profile = find_farmer(farmer_db, current_entered_name)
                if profile:
                    st.session_state.current_farmer_profile = profile; st.session_state.show_new_profile_form = False
                    loaded_lat = profile.get('latitude', PROFILE_DEFAULT_LAT); loaded_lon = profile.get('longitude', PROFILE_DEFAULT_LON)
                    st.session_state.new_profile_coords = {'lat': loaded_lat, 'lon': loaded_lon}
                    st.session_state.map_center = [loaded_lat, loaded_lon] if loaded_lat != 0.0 or loaded_lon != 0.0 else [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]
                    st.session_state.map_zoom = 12 if loaded_lat != 0.0 or loaded_lon != 0.0 else 5
                    pref_lang = profile.get('language', 'English');
                    if pref_lang not in language_options: pref_lang = "English"
                    if st.session_state.selected_language != pref_lang: st.session_state.selected_language = pref_lang
                    st.session_state.params_processed = False; st.session_state.geolocation_fetched_successfully = False
                    st.success(ui_translator("profile_loaded_success", name=profile['name'])); logger.info(f"Profile loaded for '{profile['name']}'")
                    st.rerun()
                else: st.warning(ui_translator("profile_not_found_warning", name=current_entered_name)); st.session_state.current_farmer_profile = None; st.session_state.show_new_profile_form = False
            else: st.warning(ui_translator("name_missing_error"))
        if new_button_clicked:
            if current_entered_name:
                profile = find_farmer(farmer_db, current_entered_name)
                if profile: # Profile exists, load it
                    st.toast(ui_translator("profile_exists_warning", name=current_entered_name), icon="⚠️"); st.session_state.current_farmer_profile = profile; st.session_state.show_new_profile_form = False
                    loaded_lat = profile.get('latitude', PROFILE_DEFAULT_LAT); loaded_lon = profile.get('longitude', PROFILE_DEFAULT_LON); st.session_state.new_profile_coords = {'lat': loaded_lat, 'lon': loaded_lon}
                    st.session_state.map_center = [loaded_lat, loaded_lon] if loaded_lat != 0.0 or loaded_lon != 0.0 else [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]
                    st.session_state.map_zoom = 12 if loaded_lat != 0.0 or loaded_lon != 0.0 else 5
                    pref_lang = profile.get('language', 'English');
                    if pref_lang not in language_options: pref_lang = "English"
                    if st.session_state.selected_language != pref_lang: st.session_state.selected_language = pref_lang
                    st.session_state.params_processed = False; st.session_state.geolocation_fetched_successfully = False
                    logger.info(f"Existing profile found for '{profile['name']}'. Loading."); st.rerun()
                else: # Show new profile form
                    st.info(ui_translator("creating_profile_info", name=current_entered_name)); st.session_state.show_new_profile_form = True; st.session_state.current_farmer_profile = None
                    st.session_state.new_profile_location_method = "map"; st.session_state.map_interaction_mode = "pick"
                    st.session_state.new_profile_coords = {'lat': PROFILE_DEFAULT_LAT, 'lon': PROFILE_DEFAULT_LON}
                    st.session_state.map_center = [MAP_DEFAULT_LAT, MAP_DEFAULT_LON]; st.session_state.map_zoom = 5
                    st.session_state.params_processed = False; st.session_state.geolocation_fetched_successfully = False
                    st.rerun()
            else: st.warning(ui_translator("name_missing_error"))
        st.divider()

        # --- New Profile Section (Conditional Display) ---
        form_header_name = current_entered_name if st.session_state.show_new_profile_form else ""
        if st.session_state.show_new_profile_form and form_header_name:
             # Location Method Radio
             st.subheader(ui_translator("new_profile_form_header", name=form_header_name))
             loc_method_map_text = ui_translator('loc_method_map'); loc_method_current_text = ui_translator('loc_method_current')
             loc_method_options_values = ["map", "current"]
             try: loc_method_index = loc_method_options_values.index(st.session_state.new_profile_location_method)
             except ValueError: loc_method_index = 0; st.session_state.new_profile_location_method = "map"
             def on_loc_method_change():
                 st.session_state.new_profile_location_method=st.session_state.widget_loc_method_radio
                 st.session_state.params_processed = False
                 st.session_state.geolocation_fetched_successfully = False
             st.radio(ui_translator("location_method_label"), options=loc_method_options_values, index=loc_method_index, key="widget_loc_method_radio", format_func=lambda x: loc_method_map_text if x=="map" else loc_method_current_text, on_change=on_loc_method_change)
             current_loc_method = st.session_state.new_profile_location_method

             # UI Based on Location Method
             if current_loc_method == "map":
                 st.info(ui_translator("map_instructions")); map_mode_pick_text = ui_translator('map_mode_pick'); map_mode_pan_text = ui_translator('map_mode_pan')
                 map_modes_values = ["pick", "pan"];
                 try: map_mode_index = map_modes_values.index(st.session_state.map_interaction_mode)
                 except ValueError: map_mode_index = 0; st.session_state.map_interaction_mode = "pick"
                 def on_map_mode_change(): st.session_state.map_interaction_mode=st.session_state.widget_map_mode_radio
                 st.radio(ui_translator("map_interaction_mode_label"), options=map_modes_values, index=map_mode_index, key="widget_map_mode_radio", horizontal=True, format_func=lambda x: map_mode_pick_text if x == "pick" else map_mode_pan_text, on_change=on_map_mode_change)
                 m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
                 if st.session_state.new_profile_coords['lat'] != 0.0 or st.session_state.new_profile_coords['lon'] != 0.0: folium.Marker([st.session_state.new_profile_coords['lat'], st.session_state.new_profile_coords['lon']], popup="Selected Location", tooltip="Selected Location").add_to(m)
                 map_data = st_folium(m, center=st.session_state.map_center, zoom=st.session_state.map_zoom, width=700, height=400, key="folium_map_interactive", returned_objects=[])
                 if map_data: # Process map interactions
                     new_center = map_data.get("center", st.session_state.map_center); new_zoom = map_data.get("zoom", st.session_state.map_zoom)
                     if new_center != st.session_state.map_center: st.session_state.map_center = new_center
                     if new_zoom != st.session_state.map_zoom: st.session_state.map_zoom = new_zoom
                     if map_data.get("last_clicked"):
                         clicked_lat = map_data["last_clicked"]["lat"]; clicked_lon = map_data["last_clicked"]["lng"]
                         if st.session_state.map_interaction_mode == "pick":
                             if (clicked_lat != st.session_state.new_profile_coords['lat'] or clicked_lon != st.session_state.new_profile_coords['lon']):
                                 logger.info(f"Map Pick event: Lat={clicked_lat:.4f}, Lon={clicked_lon:.4f}")
                                 st.session_state.new_profile_coords = {'lat': clicked_lat, 'lon': clicked_lon}; st.session_state.map_center = [clicked_lat, clicked_lon]; st.session_state.geolocation_fetched_successfully = False
                                 st.rerun()
                         else: logger.debug("Map clicked in Pan mode. Ignored.")
                 # Display coords
                 lat_display = st.session_state.new_profile_coords['lat']; lon_display = st.session_state.new_profile_coords['lon']
                 coord_text = f"Lat: `{lat_display:.4f}`, Lon: `{lon_display:.4f}`";
                 if lat_display == 0.0 and lon_display == 0.0: coord_text = "`Location not set. Click map to select.`"
                 st.write(f"**{ui_translator('selected_coords_label')}** {coord_text}")

             elif current_loc_method == "current": # Geolocation UI
                 lat_display = st.session_state.new_profile_coords['lat']; lon_display = st.session_state.new_profile_coords['lon']
                 if st.session_state.geolocation_fetched_successfully: st.success(ui_translator("geolocation_success_message"))
                 coord_text = f"Lat: `{lat_display:.4f}`, Lon: `{lon_display:.4f}`"
                 if lat_display == 0.0 and lon_display == 0.0 and not st.session_state.geolocation_fetched_successfully: coord_text = "`Click button below to get location.`"
                 st.write(f"**{ui_translator('selected_coords_label')}** {coord_text}")
                 st.caption(ui_translator("geolocation_info"))
                 button_label = ui_translator("get_current_loc_again_button") if st.session_state.geolocation_fetched_successfully else ui_translator("get_current_loc_button")
                 err_unavail = ui_translator("geolocation_unavailable").replace("'", "\\'"); err_denied = ui_translator("geolocation_permission_denied").replace("'", "\\'"); err_timeout = ui_translator("geolocation_timeout").replace("'", "\\'")
                 js_code = f""" <script> function getLocationAndReload() {{ const button = document.getElementById("getLocBtn"); if (!button) return; button.disabled = true; button.innerText = "Getting Location..."; if (navigator.geolocation) {{ navigator.geolocation.getCurrentPosition( (position) => {{ const lat = position.coords.latitude; const lon = position.coords.longitude; const params = new URLSearchParams(window.location.search); params.set('lat', lat.toFixed(6)); params.set('lon', lon.toFixed(6)); const newUrl = window.location.origin + window.location.pathname + '?' + params.toString(); window.location.replace(newUrl); }}, (error) => {{ let errorMsg = ""; switch(error.code) {{ case error.PERMISSION_DENIED: errorMsg = "{err_denied}"; break; case error.POSITION_UNAVAILABLE: errorMsg = "{err_unavail}"; break; case error.TIMEOUT: errorMsg = "{err_timeout}"; break; default: errorMsg = "{err_unavail}" + " (Code: " + error.code + ")"; break; }} console.error("Geolocation Error:", errorMsg, error); alert("Geolocation Error: " + errorMsg); button.disabled = false; button.innerText = "{button_label}"; }}, {{ enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }} ); }} else {{ alert("{err_unavail}"); button.disabled = false; button.innerText = "{button_label}"; }} }} </script> <button id="getLocBtn" onclick="getLocationAndReload()">{button_label}</button> """
                 components.html(js_code, height=50)

             # --- Form for Other Details (and Save Button) ---
             with st.form("new_profile_details_form", clear_on_submit=False):
                # Form fields (lang, soil, size)
                try: default_lang_index = language_options.index(st.session_state.selected_language)
                except ValueError: default_lang_index = 0
                st.selectbox(ui_translator("pref_lang_label"), options=language_options, index=default_lang_index, key="form_new_lang")
                try: default_soil_index = SOIL_TYPES.index('Unknown')
                except ValueError: default_soil_index = 0
                st.selectbox(ui_translator("soil_type_label"), options=SOIL_TYPES, index=default_soil_index, key="form_new_soil")
                st.number_input(ui_translator("farm_size_label"), value=1.0, min_value=0.01, step=0.1, format="%.2f", key="form_new_size")

                # Save Button
                submitted = st.form_submit_button(ui_translator("save_profile_button"))

                # FORM SUBMISSION LOGIC
                if submitted:
                    profile_name_to_save = form_header_name
                    if not profile_name_to_save: st.error(ui_translator("name_missing_error"))
                    else:
                        # *** Read coordinates from SESSION STATE ***
                        final_lat = st.session_state.new_profile_coords['lat']
                        final_lon = st.session_state.new_profile_coords['lon']

                        # Create data dict from form and session state coords
                        new_profile_data = {
                            'name': profile_name_to_save,
                            'language': st.session_state.form_new_lang, # Read from form state
                            'latitude': final_lat,   # Read from session state
                            'longitude': final_lon,  # Read from session state
                            'soil_type': st.session_state.form_new_soil, # Read from form state
                            'farm_size_ha': st.session_state.form_new_size # Read from form state
                        }

                        logger.info(f"--- Submitting New Profile --- Name: {profile_name_to_save}")
                        logger.info(f"Coords read from session state for save: Lat={final_lat}, Lon={final_lon}")
                        logger.info(f"Full data prepared for add/update: {new_profile_data}")

                        # Load, Update, Save DB (using modified functions)
                        current_db_state = load_or_create_farmer_db()
                        updated_db = add_or_update_farmer(current_db_state, new_profile_data)
                        if isinstance(updated_db, pd.DataFrame):
                            save_farmer_db(updated_db)
                            saved_profile = find_farmer(updated_db, profile_name_to_save)
                            if saved_profile:
                                # Update state and rerun
                                st.session_state.current_farmer_profile = saved_profile
                                st.session_state.show_new_profile_form = False # Hide form
                                st.session_state.geolocation_fetched_successfully = False # Reset flag
                                if st.session_state.selected_language != saved_profile.get('language', 'English'): st.session_state.selected_language = saved_profile.get('language', 'English')
                                st.success(ui_translator("profile_saved_success", name=profile_name_to_save))
                                logger.info(f"New profile saved successfully for '{profile_name_to_save}'. Triggering rerun.")
                                st.rerun()
                            else:
                                logger.error("Profile not found immediately after saving.")
                                st.error("Internal error: Could not reload profile after saving.")
                        else:
                            logger.error("add_or_update_farmer did not return a valid DataFrame.")
                            st.error("Internal error: Failed to update DB structure during save.")

        # Display Active Profile
        if not st.session_state.show_new_profile_form:
            st.markdown("---")
            active_profile = st.session_state.current_farmer_profile
            if active_profile and isinstance(active_profile, dict):
                st.subheader(ui_translator("active_profile_header"))
                name_disp = active_profile.get('name', 'N/A'); lang_disp = active_profile.get('language', 'N/A')
                lat_val = active_profile.get('latitude'); lon_val = active_profile.get('longitude')
                soil_disp = active_profile.get('soil_type', 'N/A'); size_val = active_profile.get('farm_size_ha')
                # Check if location is the default (0,0) before formatting
                # FIX: Call ui_translator correctly (only key and optional kwargs)
                loc_str = f"{lat_val:.4f}, {lon_val:.4f}" if (lat_val != 0.0 or lon_val != 0.0) else ui_translator('location_not_set_description')
                size_str = f"{size_val:.2f}" if isinstance(size_val, (int, float)) else 'Not Set'
                st.write(f"**{ui_translator('active_profile_name')}:** {name_disp}")
                st.write(f"**{ui_translator('active_profile_lang')}:** {lang_disp}")
                st.write(f"**{ui_translator('active_profile_loc')}:** {loc_str}")
                st.write(f"**{ui_translator('active_profile_soil')}:** {soil_disp}")
                st.write(f"**{ui_translator('active_profile_size')}:** {size_str}")
            else:
                st.info(ui_translator("no_profile_loaded_info"))


    # ================== MAIN CONTENT AREA ==================
    # Use the same ui_translator lambda defined earlier, as it references the correct current_lang_dict
    main_translator = ui_translator

    # Page Title, Header, Input Area
    st.title(main_translator("page_title")); st.caption(main_translator("page_caption")); st.divider()
    st.header(main_translator("main_header"))
    query = st.text_area(main_translator("query_label"), height=100, key="main_query_input")
    submit_button_clicked = st.button(main_translator("get_advice_button"), type="primary", key="main_submit_button")
    st.divider()

    # Processing and Output Logic
    if submit_button_clicked:
        output_lang = st.session_state.selected_language
        # Get the correct dictionary for the selected output language
        submit_time_lang_dict = translations.get(output_lang, translations["English"])
        # Create a translator function SPECIFICALLY for this submission context
        submit_translator_func = lambda k, **kwargs: t(k, submit_time_lang_dict, **kwargs)

        # Input validations
        profile_loaded = st.session_state.current_farmer_profile is not None
        query_entered = bool(st.session_state.main_query_input and st.session_state.main_query_input.strip())
        gemini_key_present = bool(st.session_state.widget_gemini_key and st.session_state.widget_gemini_key.strip())
        current_weather_key = st.session_state.widget_weather_key # May be empty

        if not profile_loaded: st.error(submit_translator_func("profile_error"))
        elif not query_entered: st.warning(submit_translator_func("query_warning"))
        elif not gemini_key_present: st.error(submit_translator_func("gemini_key_error"))
        else:
            # Proceed if prerequisites met
            current_gemini_key = st.session_state.widget_gemini_key
            llm = initialize_llm(current_gemini_key)
            if llm:
                with st.spinner(submit_translator_func("thinking_spinner", lang=output_lang)):
                    try:
                        # Call main processing function
                        result = process_farmer_request(
                            farmer_profile=st.session_state.current_farmer_profile,
                            text_query=st.session_state.main_query_input.strip(),
                            llm=llm,
                            weather_api_key=current_weather_key,
                            output_language=output_lang,
                            translator_func=submit_translator_func # Use output language translator
                        )
                        # Display results
                        farmer_display_name = result.get('farmer_name', 'Farmer')
                        if result.get("status") == "success":
                            st.header(submit_translator_func("advice_header", name=farmer_display_name, lang=output_lang))
                            st.markdown(result.get('response_text', ''))
                        else: # Display error from processing
                            st.error(result.get('response_text', submit_translator_func("processing_error", e="Unknown processing error")))
                        # Debug Expander
                        with st.expander(submit_translator_func("debug_expander_title")):
                             debug_prompt = result.get('debug_internal_prompt', submit_translator_func("debug_prompt_na"))
                             display_prompt = (debug_prompt[:2000] + '\n...') if len(debug_prompt) > 2000 else debug_prompt # Show more debug
                             st.text_area("Internal Prompt Sent to LLM (Truncated)", value=display_prompt, height=350, disabled=True, key="debug_prompt_display_area")
                    except Exception as e:
                        logger.exception("Critical error occurred in main UI processing block during request handling.")
                        st.error(submit_translator_func("processing_error", e=repr(e)))
            else: pass # Error already shown by initialize_llm

# --- Entry Point ---
if __name__ == "__main__":
    # Consider setting log level higher for production if needed
    logger.info("Starting Krishi-Sahayak AI Streamlit App...")
    main()