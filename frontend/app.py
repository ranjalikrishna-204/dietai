"""
Streamlit dashboard - the user-facing app.

Run with (from the project root `dietai/`):
    streamlit run frontend/app.py

Talks to the FastAPI backend over HTTP (config.API_URL, default http://127.0.0.1:8000) -
start the backend first: `uvicorn backend.main:app --reload`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
import plotly.graph_objects as go

import config

API = config.API_URL
CONDITIONS = ["diabetes", "hypertension", "obesity", "heart_disease", "kidney_disease", "pcos"]
VERDICT_COLOR = {"ok": "🟢", "caution": "🟡", "avoid": "🔴"}

st.set_page_config(page_title="AI Dietary Recommendation System", page_icon="🥗", layout="wide")


def api_get(path, **kw):
    r = requests.get(f"{API}{path}", timeout=15, **kw)
    r.raise_for_status()
    return r.json()


def api_post(path, **kw):
    r = requests.post(f"{API}{path}", timeout=30, **kw)
    if not r.ok:
        st.error(f"API error: {r.status_code} - {r.json().get('detail', r.text)}")
        st.stop()
    return r.json()


# ----------------------------------------------------------------------------
# Session / profile
# ----------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

st.sidebar.title("🥗 Your Profile")

if st.session_state.user_id is None:
    with st.sidebar.form("profile_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120, 25)
        sex = st.selectbox("Sex", ["male", "female", "other"])
        height_cm = st.number_input("Height (cm)", 50.0, 250.0, 165.0)
        weight_kg = st.number_input("Weight (kg)", 10.0, 400.0, 60.0)
        conditions = st.multiselect("Health conditions", CONDITIONS)
        diet_preference = st.selectbox("Diet preference", ["none", "vegetarian", "vegan", "eggetarian"])
        activity_level = st.selectbox("Activity level", ["sedentary", "light", "moderate", "active", "very_active"], index=2)
        allergies = st.text_input("Allergies (comma-separated, e.g. dairy, gluten, nuts)")
        submitted = st.form_submit_button("Create profile")
    if submitted:
        if not name:
            st.sidebar.error("Please enter your name.")
        else:
            payload = {
                "name": name, "age": age, "sex": sex, "height_cm": height_cm, "weight_kg": weight_kg,
                "conditions": conditions, "diet_preference": diet_preference, "activity_level": activity_level,
                "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
            }
            try:
                user = api_post("/users", json=payload)
            except requests.exceptions.ConnectionError:
                st.sidebar.error(f"Cannot reach the backend at {API}. Start it with:\n\nuvicorn backend.main:app --reload")
                st.stop()
            st.session_state.user_id = user["user_id"]
            st.session_state.user = user
            st.rerun()
    st.info("👈 Create a profile in the sidebar to get started.")
    st.stop()

user = st.session_state.user
st.sidebar.success(f"Signed in as **{user['name']}**")
st.sidebar.metric("BMI", f"{user['bmi']} ({user['bmi_category']})")
st.sidebar.metric("Daily calorie target", f"{user['daily_kcal_target']:.0f} kcal")
if user["conditions"]:
    st.sidebar.write("Conditions:", ", ".join(user["conditions"]))
if st.sidebar.button("Switch user"):
    st.session_state.user_id = None
    st.rerun()

tab_analyze, tab_dashboard = st.tabs(["📷 Analyze a meal", "📊 Dietary analytics"])

# ----------------------------------------------------------------------------
# Tab 1: upload & analyze
# ----------------------------------------------------------------------------
with tab_analyze:
    st.header("Upload a food photo")
    uploaded = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"])

    override_food = None
    if "all_foods" not in st.session_state:
        try:
            st.session_state.all_foods = api_get("/foods")
        except Exception:
            st.session_state.all_foods = []

    if uploaded:
        col_img, col_result = st.columns([1, 2])
        with col_img:
            st.image(uploaded, caption="Uploaded photo", use_container_width=True)
            with st.expander("Not recognised correctly? Pick the food manually"):
                override_food = st.selectbox("Food", ["(auto-detect)"] + st.session_state.all_foods)
                manual_portion = st.number_input("Portion (grams)", 10, 2000, 150)

        with col_result:
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"user_id": user["user_id"]}
            if override_food and override_food != "(auto-detect)":
                data["manual_food"] = override_food
                data["manual_portion_g"] = manual_portion

            with st.spinner("Analyzing..."):
                result = api_post("/analyze", files=files, data=data)

            st.subheader("Detected items")
            for det in result["detections"]:
                st.write(f"**{det['display_name']}** - {det['confidence']*100:.0f}% confidence, "
                         f"~{det['portion_g']:.0f} g")

            st.subheader("Nutrition")
            n_cols = st.columns(len(result["nutrition"]) or 1)
            for col, n in zip(n_cols, result["nutrition"]):
                with col:
                    st.metric(n["display_name"], f"{n['kcal']:.0f} kcal")
                    st.caption(f"P {n['protein_g']:.1f}g · C {n['carb_g']:.1f}g · F {n['fat_g']:.1f}g · "
                               f"Sugar {n['sugar_g']:.1f}g · Na {n['sodium_mg']:.0f}mg")

            totals = result["meal_totals"]
            st.info(f"**Meal totals:** {totals['kcal']:.0f} kcal | {totals['protein_g']:.1f}g protein | "
                    f"{totals['carb_g']:.1f}g carbs | {totals['fat_g']:.1f}g fat | "
                    f"{totals['sodium_mg']:.0f}mg sodium | {totals['sugar_g']:.1f}g sugar")

            st.subheader("Health-condition suitability")
            for food, flags in result["suitability"].items():
                display = next((n["display_name"] for n in result["nutrition"] if n["food"] == food), food)
                st.markdown(f"**{display}**")
                for f in flags:
                    st.write(f"{VERDICT_COLOR.get(f['verdict'],'')} *{f['condition']}*: {'; '.join(f['reasons'])}")

            alt_any = any(result["alternatives"].values())
            if alt_any:
                st.subheader("Healthier alternatives")
                for food, alts in result["alternatives"].items():
                    if not alts:
                        continue
                    for a in alts:
                        st.write(f"🔄 Instead of **{food.replace('_',' ').title()}**, try **{a['display_name']}** ({a['why']})")

            tips_any = any(result["cooking_tips"].values())
            if tips_any:
                st.subheader("Healthy cooking guidance")
                for food, tips in result["cooking_tips"].items():
                    for t in tips:
                        st.write(f"👩‍🍳 {t}")

            if result["general_advice"]:
                with st.expander("General advice"):
                    for a in result["general_advice"]:
                        st.write(f"- {a}")

            if st.button("✅ Log this meal to today's diary"):
                foods = [n["food"] for n in result["nutrition"]]
                portions = [n["portion_g"] for n in result["nutrition"]]
                api_post("/log", json={"user_id": user["user_id"], "foods": foods, "portions_g": portions})
                st.success("Logged!")

# ----------------------------------------------------------------------------
# Tab 2: dashboard
# ----------------------------------------------------------------------------
with tab_dashboard:
    st.header("Today's intake vs. target")
    daily = api_get(f"/analytics/{user['user_id']}/daily")
    target = user["daily_kcal_target"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=daily["kcal"],
        title={"text": "Calories (kcal)"},
        gauge={"axis": {"range": [0, max(target * 1.3, daily["kcal"] * 1.1, 100)]},
               "steps": [{"range": [0, target], "color": "#d4edda"},
                         {"range": [target, target * 1.3], "color": "#f8d7da"}],
               "threshold": {"line": {"color": "red", "width": 3}, "value": target}},
    ))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Protein", f"{daily['protein_g']:.0f} g")
    c2.metric("Carbs", f"{daily['carb_g']:.0f} g")
    c3.metric("Fat", f"{daily['fat_g']:.0f} g")
    c4.metric("Sodium", f"{daily['sodium_mg']:.0f} mg")

    st.header("Last 14 days")
    hist = api_get(f"/analytics/{user['user_id']}/history", params={"days": 14})
    if hist:
        import pandas as pd
        df = pd.DataFrame(hist)
        daily_kcal = df.groupby("date")["kcal"].sum().reset_index()
        fig2 = go.Figure(go.Bar(x=daily_kcal["date"], y=daily_kcal["kcal"]))
        fig2.add_hline(y=target, line_dash="dash", line_color="red", annotation_text="daily target")
        fig2.update_layout(yaxis_title="kcal")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Raw log"):
            st.dataframe(df, use_container_width=True)
    else:
        st.write("No meals logged yet - analyze a photo and click **Log this meal** to start tracking.")
