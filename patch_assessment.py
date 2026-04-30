with open('assistant_core.py', 'r', encoding='utf-8') as f:
    code = f.read()

assessment_method = """    def _generate_assessment(self, profile: dict) -> str:
        w = profile.get("weight")
        h = profile.get("height")
        if not w or not h:
            return ""
        
        gender = profile.get("gender", "Male").lower()
        
        # BMI Calculation
        height_m = h / 100
        bmi = w / (height_m * height_m)
        
        # BMR (Mifflin-St Jeor) - assuming age 30
        if gender == "female":
            bmr = 10 * w + 6.25 * h - 5 * 30 - 161
        else:
            bmr = 10 * w + 6.25 * h - 5 * 30 + 5
            
        tdee = bmr * 1.55 # moderate activity
        
        goal = profile.get("goal", "fat_burn")
        if goal == "muscle_gain":
            target_cal = tdee + 400
        else:
            target_cal = tdee - 500
            
        lines = [
            f"🎯 Accurate Physiological Assessment:",
            f"• Body Mass Index (BMI): {bmi:.1f} ({profile.get('bmi_category', 'Calculated')})",
            f"• Basal Metabolic Rate (BMR): {int(bmr)} kcal/day",
            f"• Total Daily Energy Expenditure (TDEE): ~{int(tdee)} kcal/day",
            f"• Daily Target Calories for Goal: {int(target_cal)} kcal/day",
            f"• Recommended Protein Intake: ~{int(w * 2.2)}g per day"
        ]
        
        # Add body measurements if provided
        chest, waist, hips = profile.get("chest"), profile.get("waist"), profile.get("hips")
        if chest or waist or hips:
            lines.append(f"• Measurements Logged: Chest {chest or '-'}cm, Waist {waist or '-'}cm, Hips {hips or '-'}cm")
        
        biceps, body_fat = profile.get("biceps"), profile.get("body_fat")
        if biceps:
            lines.append(f"• Biceps Logged: {biceps}cm")
        if body_fat:
            lines.append(f"• Current Body Fat: {body_fat}%")
            
        med = profile.get("medical_history")
        if med and str(med).strip().lower() not in ["none", "no", "n/a", ""]:
            lines.append(f"⚠️ Medical Note: Acknowledged ({med}). Please ensure workouts align with your physician's advice.")
            
        return "\\n".join(lines)

    def _answer_topic(self, message: str, profile: dict) -> dict:"""

code = code.replace("    def _answer_topic(self, message: str, profile: dict) -> dict:", assessment_method)

# Inject in recommend
recommend_new = """        nutrition_examples = self._nutrition_picks(profile)
        assessment_text = self._generate_assessment(profile)
        reply = ("""

recommend_old = """        nutrition_examples = self._nutrition_picks(profile)
        reply = ("""

code = code.replace(recommend_old, recommend_new)

ret_new = """            "recommendation": recommendation_to_dict(recommendation),
            "assessment": assessment_text,
            "parsed": parsed,"""

ret_old = """            "recommendation": recommendation_to_dict(recommendation),
            "parsed": parsed,"""

code = code.replace(ret_old, ret_new)

with open('assistant_core.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Math assessment injected.")
