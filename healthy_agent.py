# ============================================================
#  Healthy Habits Agent
#  Stack : FastAPI + Uvicorn + ngrok + LangChain + gpt-4o-mini
#  Tools : BMI Calculator, Calorie/TDEE Estimator,
#          Ideal Weight Calculator, Water Intake Recommender
#  Tests : Automated HTTP tests that run after server starts
#
#  INSTALL (run once before starting):
#  pip install -r requirements.txt
# ============================================================

# --------------- standard library ---------------
import os
import sys
import math
import time
import threading

# --------------- third-party --------------------
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok, conf as ngrok_conf

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# ============================================================
# 0.  Load environment variables
# ============================================================
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

if not OPENAI_API_KEY:
    sys.exit("ERROR: OPENAI_API_KEY not found in .env file.")
if not NGROK_AUTH_TOKEN:
    sys.exit("ERROR: NGROK_AUTH_TOKEN not found in .env file.")

# Set via environment so LangChain picks it up without any
# constructor kwarg — avoids pydantic 'proxies' ValidationError.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ============================================================
# 1.  Domain Tools
# ============================================================

@tool
def calculate_bmi(weight_kg: float, height_cm: float, age: int) -> str:
    """
    Calculate BMI given weight in kilograms, height in centimetres, and age in years.
    Returns BMI value and WHO category with age-contextual note.

    Args:
        weight_kg: Body weight in kilograms (e.g. 70)
        height_cm: Height in centimetres (e.g. 175)
        age: Age in years (e.g. 30)
    """
    if weight_kg <= 0 or height_cm <= 0 or age <= 0:
        return "Invalid input: weight, height, and age must all be positive numbers."

    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        category = "Underweight"
        advice   = "Consider increasing caloric intake with nutrient-dense foods."
    elif bmi < 25.0:
        category = "Normal weight"
        advice   = "Great! Maintain your current lifestyle with balanced diet and exercise."
    elif bmi < 30.0:
        category = "Overweight"
        advice   = "Consider moderate calorie reduction and increasing physical activity."
    else:
        category = "Obese"
        advice   = "Consulting a healthcare professional for a personalised plan is recommended."

    age_note = ""
    if age >= 65:
        age_note = " Note: For adults 65+, a slightly higher BMI (23-27) may be protective."
    elif age < 18:
        age_note = " Note: BMI interpretation for children/teens requires age-specific growth charts."

    return (
        f"BMI: {bmi}\n"
        f"Category: {category}\n"
        f"Advice: {advice}{age_note}"
    )


@tool
def estimate_daily_calories(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str,
) -> str:
    """
    Estimate Total Daily Energy Expenditure (TDEE) using Mifflin-St Jeor equation.

    Args:
        weight_kg: Body weight in kilograms
        height_cm: Height in centimetres
        age: Age in years
        gender: 'male' or 'female'
        activity_level: One of 'sedentary', 'light', 'moderate', 'active', 'very_active'
    """
    gender         = gender.lower().strip()
    activity_level = activity_level.lower().strip()

    if gender not in ("male", "female"):
        return "Gender must be 'male' or 'female'."

    activity_map = {
        "sedentary":   1.2,
        "light":       1.375,
        "moderate":    1.55,
        "active":      1.725,
        "very_active": 1.9,
    }
    if activity_level not in activity_map:
        return "activity_level must be one of: sedentary, light, moderate, active, very_active."

    if gender == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    tdee = round(bmr * activity_map[activity_level], 0)
    loss = round(tdee - 500, 0)
    gain = round(tdee + 300, 0)

    return (
        f"Basal Metabolic Rate (BMR): {round(bmr)} kcal/day\n"
        f"Total Daily Energy Expenditure (TDEE): {tdee} kcal/day\n"
        f"For weight loss (~0.5 kg/week): {loss} kcal/day\n"
        f"For muscle gain: {gain} kcal/day\n"
        f"Activity level used: {activity_level}"
    )


@tool
def calculate_ideal_weight(height_cm: float, gender: str) -> str:
    """
    Calculate ideal body weight range using multiple formulas
    (Devine, Robinson, Miller, Hamwi).

    Args:
        height_cm: Height in centimetres
        gender: 'male' or 'female'
    """
    gender = gender.lower().strip()
    if gender not in ("male", "female"):
        return "Gender must be 'male' or 'female'."
    if height_cm <= 0:
        return "Height must be a positive number."

    height_in       = height_cm / 2.54
    inches_over_5ft = max(0, height_in - 60)

    if gender == "male":
        devine   = 50.0 + 2.3  * inches_over_5ft
        robinson = 52.0 + 1.9  * inches_over_5ft
        miller   = 56.2 + 1.41 * inches_over_5ft
        hamwi    = 48.0 + 2.7  * inches_over_5ft
    else:
        devine   = 45.5 + 2.3  * inches_over_5ft
        robinson = 49.0 + 1.7  * inches_over_5ft
        miller   = 53.1 + 1.36 * inches_over_5ft
        hamwi    = 45.5 + 2.2  * inches_over_5ft

    weights = [devine, robinson, miller, hamwi]
    avg  = round(sum(weights) / len(weights), 1)
    low  = round(min(weights), 1)
    high = round(max(weights), 1)

    return (
        f"Ideal Weight Estimates for {gender}, height {height_cm} cm:\n"
        f"  Devine formula  : {round(devine,  1)} kg\n"
        f"  Robinson formula: {round(robinson,1)} kg\n"
        f"  Miller formula  : {round(miller,  1)} kg\n"
        f"  Hamwi formula   : {round(hamwi,   1)} kg\n"
        f"Average ideal weight : {avg} kg\n"
        f"Healthy range    : {low} - {high} kg\n"
        f"Note: These are statistical guidelines; individual health goals may differ."
    )


@tool
def recommend_water_intake(
    weight_kg: float,
    activity_level: str,
    climate: str = "moderate",
) -> str:
    """
    Recommend daily water intake based on body weight, activity level, and climate.

    Args:
        weight_kg: Body weight in kilograms
        activity_level: One of 'sedentary', 'light', 'moderate', 'active', 'very_active'
        climate: 'cool', 'moderate', or 'hot'
    """
    if weight_kg <= 0:
        return "Weight must be a positive number."

    activity_level = activity_level.lower().strip()
    climate        = climate.lower().strip()

    activity_add = {
        "sedentary":   0.0,
        "light":       0.35,
        "moderate":    0.5,
        "active":      0.75,
        "very_active": 1.0,
    }
    climate_add = {"cool": 0.0, "moderate": 0.25, "hot": 0.5}

    if activity_level not in activity_add:
        return "activity_level must be one of: sedentary, light, moderate, active, very_active."
    if climate not in climate_add:
        return "climate must be one of: cool, moderate, hot."

    base_litres  = weight_kg * 0.035
    total_litres = round(base_litres + activity_add[activity_level] + climate_add[climate], 2)
    glasses      = math.ceil(total_litres / 0.25)

    return (
        f"Recommended daily water intake:\n"
        f"  Body weight   : {weight_kg} kg\n"
        f"  Activity level: {activity_level}\n"
        f"  Climate       : {climate}\n"
        f"  Total         : {total_litres} litres/day (~{glasses} glasses of 250 ml)\n"
        f"Tip: Spread intake throughout the day; drink more during exercise and in heat."
    )


# ============================================================
# 2.  LangChain Agent
# ============================================================

SYSTEM_PROMPT = """
You are a Healthy Habits Agent — a knowledgeable, friendly assistant
specialising EXCLUSIVELY in healthy habits, wellness, nutrition,
physical fitness, sleep, hydration, and general preventive health.

You have access to four tools:
  * calculate_bmi           - computes BMI and WHO category
  * estimate_daily_calories - estimates TDEE using Mifflin-St Jeor
  * calculate_ideal_weight  - ideal weight via four clinical formulas
  * recommend_water_intake  - daily hydration target

Rules:
1. ONLY answer questions related to healthy habits, nutrition,
   fitness, hydration, sleep, or wellness.
2. If a question is outside this domain, respond EXACTLY with:
   "Don't know, suggest user to ask questions around health."
3. Always use the appropriate tool when numerical health metrics
   are involved. Do NOT guess numbers.
4. Be concise, warm, and evidence-based.
5. Always remind users to consult a healthcare professional for
   medical diagnoses or treatment.
"""


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    tools = [
        calculate_bmi,
        estimate_daily_calories,
        calculate_ideal_weight,
        recommend_water_intake,
    ]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor = build_agent()

# ============================================================
# 3.  FastAPI Application
# ============================================================

app = FastAPI(
    title="Healthy Habits Agent API",
    description="AI-powered agent for BMI, calories, ideal weight & hydration.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Healthy Habits Agent is live!",
        "endpoints": {
            "POST /ask":    "Ask a healthy-habits question",
            "GET  /health": "Server health check",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "ready"}


@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = agent_executor.invoke({"input": request.question})
        answer = result.get("output", "No response generated.")
        return QueryResponse(question=request.question, answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# 4.  ngrok tunnel launcher
# ============================================================

def start_ngrok(port: int) -> str:
    ngrok_conf.get_default().auth_token = NGROK_AUTH_TOKEN
    tunnel     = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    print(f"\n{'='*55}")
    print(f"  ngrok public URL : {public_url}")
    print(f"  Local server     : http://127.0.0.1:{port}")
    print(f"  Swagger UI       : {public_url}/docs")
    print(f"{'='*55}\n")
    return public_url


# ============================================================
# 5.  Automated Test Suite
# ============================================================

TEST_CASES = [
    (
        "BMI calculation for a 30-year-old male",
        {"question": "Calculate my BMI. I weigh 80 kg, I am 175 cm tall and 30 years old."},
        "bmi",
        True,
    ),
    (
        "Calorie estimation - sedentary female",
        {"question": "Estimate daily calories for a 28-year-old female, 60 kg, 165 cm, sedentary."},
        "kcal",
        True,
    ),
    (
        "Ideal weight for a 180 cm male",
        {"question": "What is the ideal weight for a male who is 180 cm tall?"},
        "kg",
        True,
    ),
    (
        "Water intake - very active, hot climate",
        {"question": "How much water should I drink? I weigh 75 kg, very active, hot climate."},
        "litre",
        True,
    ),
    (
        "Out-of-domain - stock market",
        {"question": "What are the best tech stocks to buy right now?"},
        "don't know",
        True,
    ),
    (
        "Out-of-domain - recipe",
        {"question": "Give me a chocolate cake recipe."},
        "don't know",
        True,
    ),
    (
        "In-domain - sleep hours",
        {"question": "How many hours of sleep does an adult need per night?"},
        "",
        True,
    ),
    (
        "Empty question - expect HTTP 400",
        {"question": ""},
        "400",
        False,
    ),
]


def run_tests(base_url: str):
    print("\n" + "="*55)
    print("  Running Automated Test Suite")
    print("="*55)

    passed = 0
    failed = 0

    for idx, (desc, payload, expected, should_pass) in enumerate(TEST_CASES, 1):
        print(f"\n[Test {idx}] {desc}")
        print(f"  Question : {payload['question'] or '(empty)'}")
        try:
            resp = requests.post(
                f"{base_url}/ask",
                json=payload,
                timeout=60,
            )
            if should_pass:
                if resp.status_code == 200:
                    answer = resp.json().get("answer", "")
                    if expected == "" or expected.lower() in answer.lower():
                        print(f"  PASSED  | Answer snippet: {answer[:120]}")
                        passed += 1
                    else:
                        print(f"  FAILED  | Expected '{expected}' in answer.")
                        print(f"           | Got: {answer[:200]}")
                        failed += 1
                else:
                    print(f"  FAILED  | HTTP {resp.status_code}: {resp.text[:200]}")
                    failed += 1
            else:
                if resp.status_code >= 400:
                    print(f"  PASSED  | Expected error, got HTTP {resp.status_code}")
                    passed += 1
                else:
                    print(f"  FAILED  | Expected error but got HTTP {resp.status_code}")
                    failed += 1
        except requests.exceptions.RequestException as exc:
            print(f"  ERROR   | {exc}")
            failed += 1

    print("\n" + "="*55)
    print(f"  Results : {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("="*55 + "\n")


# ============================================================
# 6.  Entry Point
# ============================================================

PORT = 8000


class ServerThread(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port

    def run(self):
        uvicorn.run(app, host="0.0.0.0", port=self.port, log_level="warning")


if __name__ == "__main__":
    print("\nStarting Healthy Habits Agent server ...")
    server_thread = ServerThread(PORT)
    server_thread.start()
    time.sleep(2)

    public_url = start_ngrok(PORT)

    local_url = f"http://127.0.0.1:{PORT}"
    print("Waiting 3 s for agent to warm up ...")
    time.sleep(3)

    test_thread = threading.Thread(target=run_tests, args=(local_url,), daemon=True)
    test_thread.start()
    test_thread.join()

    print("Server is still running. Press Ctrl+C to stop.\n")
    print(f"  Public URL   : {public_url}/ask")
    print(f"  Swagger docs : {public_url}/docs\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down ...")
        ngrok.disconnect(public_url)
        ngrok.kill()
        sys.exit(0)
