# Travel Planner AI Chatbot

## Overview
The **Travel Planner AI Chatbot** is a generative AI project designed to assist travelers in planning their trips.  
It uses OpenAI’s Function Calling and natural language understanding to provide **top hotel recommendations** and **custom itineraries** based on user-specified destinations.

This chatbot demonstrates how **GenAI**, **data-driven recommendations**, and **safety mechanisms** can work together to create a user-friendly and secure conversational travel assistant.

---

## Objectives
- Develop a travel assistant that helps users plan trips with AI.
- Recommend **top 3 hotels** and **custom itineraries** from a CSV dataset.
- Integrate **moderation** to detect and close harmful conversations.
- Ensure output consistency and conversational safety.
- Demonstrate full **end-to-end system design and implementation**.

---

## System Architecture
1. **User Interface Layer** – Collects user queries through chat.
2. **LLM Processing Layer** – Uses OpenAI API for understanding and reasoning.
3. **Function Calling Module** – Interprets structured output for hotel search and itinerary.
4. **Data Management Layer** – Reads `hotels.csv` containing 10K+ hotel records in INR.
5. **Moderation Module** – Detects unsafe inputs using `omni-moderation-latest`.
6. **Response Composition Layer** – Uses `compose_response_llm()` for structured output.

---

## Implementation Details
- **Language:** Python 3.x  
- **Frameworks/Libraries:** `openai`, `pandas`, `flask`, `docx`  
- **Data Source:** Publicly available dataset (`hotels.csv`) with hotel names, ratings, and prices (INR).  
- **AI Model:** GPT model for chat completion and function calling.

### Key Features:
- Reads hotel data from CSV directly.
- Returns top 3 hotels matching the destination.
- If no matches, outputs: _“No itinerary found as per your requirement.”_
- Closes session safely if harmful input is detected.
- Uses `compose_response_llm()` for clear and formatted responses.

---

## Code Snippet Example

```python
def compose_response_llm(user_query: str, hotels: list, itinerary: dict):
    system_msg = "You are a helpful travel assistant that gives concise, friendly suggestions."
    hotels_section = ""
    for i, h in enumerate(hotels, start=1):
        hotels_section += f"{i}. {h.get('name')} — {h.get('rating','')}⭐ | Price: ₹{h.get('price_per_night','n/a')}
"
    if not hotels_section:
        hotels_section = "No hotels found for this destination.
"

    itinerary_section = itinerary['itinerary_text'] if itinerary else "No itinerary found as per your requirement."
    return f"{system_msg}\n\nHotels:\n{hotels_section}\n\nItinerary:\n{itinerary_section}"
```

---

## 🧪 Challenges Faced
- Model compatibility issues (`Invalid value for model` errors).
- Integration of moderation API with LLM-based chatbot.
- Handling 10K+ dataset efficiently.
- Ensuring consistency in generated AI responses.
- Implementing safe session termination for harmful content.

---

## Lessons Learned
- Always verify model version compatibility with SDK.
- Safety and moderation are crucial in open-ended chatbots.
- Function calling significantly improves structured AI responses.
- Modular design helps maintain clarity and scalability.

---

## Conclusion
The **Travel Planner AI Chatbot** demonstrates a practical use of GenAI in travel planning.  
It combines **AI reasoning**, **data-driven recommendations**, and **safety enforcement**, making it a strong example of a hybrid, responsible AI system.

---

## Files Included
- `Travel_Planner_AI_Project.ipynb` – Notebook with code implementation  
- `hotels.csv` – Dataset with 10K+ hotel records in INR  
- `Travel_Planner_AI_Project_Report.docx` – Detailed project report  
- `README.md` – Project documentation  

---

## Author
**Deepika Ramesh**  
AI & Cloud Automation Engineer | Python | GenAI   
