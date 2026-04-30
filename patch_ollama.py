import re

with open('assistant_core.py', 'r', encoding='utf-8') as f:
    code = f.read()

llm_method = """    def _generate_llm_response(self, message: str, profile: dict, exercises: list, foods: list) -> str:
        prompt = f"You are FitPax, an elite, professional AI fitness and nutrition coach. The user says: '{message}'\\n"
        if any(profile.values()):
            prompt += f"User Profile: {profile}\\n"
        if exercises:
            ex_names = ", ".join(e.get("name", "") for e in exercises)
            prompt += f"You are recommending these exercises: {ex_names}.\\n"
        if foods:
            food_names = ", ".join(f.get("name", "") for f in foods)
            prompt += f"You are recommending these foods: {food_names}.\\n"
        prompt += "Write a highly professional, encouraging, and concise response (max 2-3 sentences) directly addressing the user. Do not use asterisks, lists, or markdown formatting. Keep it conversational and supportive."

        try:
            import urllib.request
            import json
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception:
            return ""

    def _answer_topic(self, message: str, profile: dict) -> dict:"""

code = code.replace("    def _answer_topic(self, message: str, profile: dict) -> dict:", llm_method)


# Inject in recommend
recommend_old = """        if knowledge_reply:
            reply = f"{reply} {knowledge_reply}"

        memory["profile"] = profile"""

recommend_new = """        if knowledge_reply:
            reply = f"{reply} {knowledge_reply}"

        llm_reply = self._generate_llm_response(description, profile, exercises, nutrition_examples)
        if llm_reply:
            reply = llm_reply

        memory["profile"] = profile"""

code = code.replace(recommend_old, recommend_new)

# Inject in chat question_mode
chat_q_old = """        if question_mode:
            answer = self._answer_topic(message or description, profile)
            answer.setdefault("profile", profile)"""

chat_q_new = """        if question_mode:
            answer = self._answer_topic(message or description, profile)
            llm_reply = self._generate_llm_response(message or description, profile, answer.get("exercise_examples", []), answer.get("nutrition_examples", []))
            if llm_reply:
                answer["reply"] = llm_reply
            answer.setdefault("profile", profile)"""

code = code.replace(chat_q_old, chat_q_new)


# Inject in chat fallback
chat_fb_old = """        answer = self._answer_topic(message or description, profile)
        answer.setdefault("profile", profile)"""

chat_fb_new = """        answer = self._answer_topic(message or description, profile)
        llm_reply = self._generate_llm_response(message or description, profile, answer.get("exercise_examples", []), answer.get("nutrition_examples", []))
        if llm_reply:
            answer["reply"] = llm_reply
        answer.setdefault("profile", profile)"""

code = code.replace(chat_fb_old, chat_fb_new)


with open('assistant_core.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("LLM injected.")
