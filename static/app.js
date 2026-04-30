function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const planBtn = document.getElementById("planBtn");
const messageInput = document.getElementById("messageInput");
const gender = document.getElementById("gender");
const goal = document.getElementById("goal");
const bmiCategory = document.getElementById("bmiCategory");
const dietType = document.getElementById("dietType");
const description = document.getElementById("description");
let lastAssistantPayload = null;
let sessionId = localStorage.getItem("fitpax_session_id");
if (!sessionId) {
  sessionId = "sess_" + Math.random().toString(36).substring(2, 15);
  localStorage.setItem("fitpax_session_id", sessionId);
}

const fallbackOptions = {
  gender: [
    { value: "", label: "Optional" },
    { value: "Male", label: "Male" },
    { value: "Female", label: "Female" },
  ],
  goal: [
    { value: "", label: "Optional" },
    { value: "fat_burn", label: "Fat burn" },
    { value: "muscle_gain", label: "Muscle gain" },
  ],
  bmi_category: [
    { value: "", label: "Optional" },
    { value: "Underweight", label: "Underweight" },
    { value: "Normal weight", label: "Normal weight" },
    { value: "Overweight", label: "Overweight" },
    { value: "Obesity", label: "Obesity" },
  ],
  diet_type: [
    { value: "", label: "Optional" },
    { value: "vegan", label: "Vegan" },
    { value: "vegetarian", label: "Vegetarian" },
  ],
};

function stateFromForm() {
  return {
    profile: profileStateFromForm(),
    description: description.value || "",
  };
}

function profileStateFromForm() {
  const profile = {};
  if (gender.value) profile.gender = gender.value;
  if (goal.value) profile.goal = goal.value;
  if (dietType?.value) profile.diet_type = dietType.value;
  
  const med = document.getElementById("medicalHistory")?.value;
  if (med) profile.medical_history = med;
  
  const w = parseFloat(document.getElementById("weight")?.value);
  if (!isNaN(w)) profile.weight = w;
  
  const h = parseFloat(document.getElementById("height")?.value);
  if (!isNaN(h)) profile.height = h;
  
  const c = parseFloat(document.getElementById("chest")?.value);
  if (!isNaN(c)) profile.chest = c;
  
  const wa = parseFloat(document.getElementById("waist")?.value);
  if (!isNaN(wa)) profile.waist = wa;
  
  const hi = parseFloat(document.getElementById("hips")?.value);
  if (!isNaN(hi)) profile.hips = hi;
  
  const b = parseFloat(document.getElementById("biceps")?.value);
  if (!isNaN(b)) profile.biceps = b;
  
  const bf = parseFloat(document.getElementById("bodyFat")?.value);
  if (!isNaN(bf)) profile.body_fat = bf;
  
  if (profile.height && profile.weight) {
    const heightM = profile.height / 100;
    const bmi = profile.weight / (heightM * heightM);
    if (bmi < 18.5) profile.bmi_category = "Underweight";
    else if (bmi < 25) profile.bmi_category = "Normal weight";
    else if (bmi < 30) profile.bmi_category = "Overweight";
    else profile.bmi_category = "Obesity";
  } else if (bmiCategory?.value) {
    profile.bmi_category = bmiCategory.value;
  }
  
  return profile;
}

function populateSelect(select, options, currentValue = "") {
  const selected = currentValue || select.value || "";
  select.innerHTML = "";
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value || "";
    item.textContent = option.label || option.value || "Option";
    if ((option.value || "") === selected) {
      item.selected = true;
    }
    select.appendChild(item);
  }
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  messages.appendChild(bubble);
  scrollToBottom();
}

async function sendFeedback(rating, payload) {
  try {
    await postJson("/feedback", {
      rating,
      name: payload.recommendation?.exercise_schedule || payload.recommendation?.meal_plan || payload.reply || payload.kind,
      reply: payload.reply || "",
      message: payload.profile?.goal || "",
      profile: payload.profile || {},
    });
  } catch (error) {
    // Feedback is optional. Ignore network errors silently.
  }
}

function addPlanCard(payload) {
  lastAssistantPayload = payload;
  const card = document.createElement("div");
  card.className = "bubble bot card";
  const workoutText = payload.recommendation?.exercise_schedule || "";
  const mealText = payload.recommendation?.meal_plan || "";
  const nutritionNote = payload.meal_guidance || "";
  const hasPlanDetails = workoutText || mealText || nutritionNote;
  const title = payload.kind === "memory" ? "Memory hit" : payload.kind === "answer" ? "Answer ready" : "Plan ready";
  const matchText = payload.recommendation
    ? `${Math.round((payload.recommendation?.confidence || 0) * 100)}% match`
    : "Context match";

  const inferred = Array.isArray(payload.defaults_used) && payload.defaults_used.length
    ? `<div class="hint">Inferred: ${escapeHtml(payload.defaults_used.join(", "))}</div>`
    : "";
  const grounded = Array.isArray(payload.knowledge_examples) && payload.knowledge_examples.length
    ? `<div class="hint">Grounded in ${payload.knowledge_examples.length} dataset examples.</div>`
    : "";

  const exercises = Array.isArray(payload.exercise_examples) && payload.exercise_examples.length
    ? `<div class="table-container">
        <table class="plan-table">
          <thead>
            <tr>
              <th>Image</th>
              <th>Target Muscle</th>
              <th>Exercise Name</th>
              <th>Instructions</th>
            </tr>
          </thead>
          <tbody>
            ${payload.exercise_examples.map((exercise) => `
              <tr>
                <td class="img-cell">
                  ${
                    exercise.gif_url
                      ? `<img class="table-img" src="${escapeHtml(exercise.gif_url)}" alt="${escapeHtml(exercise.name || "Exercise")}" loading="lazy" />`
                      : ""
                  }
                </td>
                <td>${escapeHtml(exercise.muscles || "Training")}</td>
                <td><strong>${escapeHtml(exercise.name || "Exercise")}</strong></td>
                <td class="instructions-cell">${escapeHtml(exercise.instruction || "")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>`
    : "";

  const nutrition = Array.isArray(payload.nutrition_examples) && payload.nutrition_examples.length
    ? `<div class="table-container">
        <table class="plan-table">
          <thead>
            <tr>
              <th>Food Item</th>
              <th>Calories (kcal)</th>
              <th>Protein</th>
              <th>Fiber</th>
            </tr>
          </thead>
          <tbody>
            ${payload.nutrition_examples.map((item) => `
              <tr>
                <td><strong>${escapeHtml(item.name || "Food")}</strong></td>
                <td>${escapeHtml(item.calories || "-")}</td>
                <td>${escapeHtml(item.protein || "-")}</td>
                <td>${escapeHtml(item.fiber || "-")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>`
    : "";

  const assessment = payload.assessment ? `<div class="result-item" style="grid-column: 1 / -1; white-space: pre-wrap;"><span>Advanced Body Assessment</span><strong>${escapeHtml(payload.assessment)}</strong></div>` : '';

  card.innerHTML = `
    <div class="card-head">
      <div>
        <div class="card-label">${escapeHtml(title)}</div>
        <div class="card-title">${escapeHtml(payload.profile?.goal || "gym")} plan</div>
      </div>
      <div class="match">${escapeHtml(matchText)}</div>
    </div>
    ${inferred}
    ${grounded}
    <div class="hint">${escapeHtml(payload.weekly_guidance || "")}</div>
    ${(hasPlanDetails || payload.assessment) ? `
    <div class="result-grid">
      ${assessment}
      ${workoutText ? `<div class="result-item"><span>Workout</span><strong>${escapeHtml(workoutText)}</strong></div>` : ''}
      ${mealText ? `<div class="result-item"><span>Meal plan</span><strong>${escapeHtml(mealText)}</strong></div>` : ''}
      ${nutritionNote ? `<div class="result-item"><span>Nutrition note</span><strong>${escapeHtml(nutritionNote)}</strong></div>` : ''}
    </div>
    ` : ''}
    ${exercises ? `
    <div class="section-label">Suggested exercises</div>
    ${exercises}
    ` : ''}
    ${nutrition ? `
    <div class="section-label">Suggested foods</div>
    ${nutrition}
    ` : ''}
    ${
      Array.isArray(payload.suggestions) && payload.suggestions.length
        ? `<div class="section-label">Next questions</div><div class="chip-row">${payload.suggestions.map((item) => `<button type="button" class="chip" data-fill="${escapeHtml(item)}">${escapeHtml(item)}</button>`).join("")}</div>`
        : ""
    }
    <div class="feedback-row">
      <button type="button" class="feedback-btn" data-rating="up">Helpful</button>
      <button type="button" class="feedback-btn" data-rating="down">Not helpful</button>
    </div>
  `;
  messages.appendChild(card);
  scrollToBottom();
}

function setPending() {
  const pending = document.createElement("div");
  pending.className = "bubble bot pending";
  pending.textContent = "Thinking...";
  messages.appendChild(pending);
  scrollToBottom();
  return pending;
}

async function postJson(path, body) {
  body.session_id = sessionId;
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  return { response, data };
}

async function loadOptions() {
  try {
    const response = await fetch("/options");
    const data = await response.json();
    if (!response.ok || !data) throw new Error("options_failed");
    populateSelect(gender, data.gender || [], gender.value);
    populateSelect(goal, data.goal || [], goal.value);
    populateSelect(bmiCategory, data.bmi_category || [], bmiCategory.value);
    if (dietType) {
      populateSelect(dietType, data.diet_type || [], dietType.value);
    }
  } catch (error) {
    populateSelect(gender, fallbackOptions.gender, gender.value);
    populateSelect(goal, fallbackOptions.goal, goal.value);
    populateSelect(bmiCategory, fallbackOptions.bmi_category, bmiCategory.value);
    if (dietType) {
      populateSelect(dietType, fallbackOptions.diet_type, dietType.value);
    }
  }
}

async function sendChat(message) {
  const text = message.trim();
  if (!text) return;
  addBubble("user", text);
  const pending = setPending();
  try {
    const payload = { ...stateFromForm(), message: text, state: { profile: profileStateFromForm() } };
    const { response, data } = await postJson("/chat", payload);
    pending.remove();
    if (!response.ok || !data.ok) {
      addBubble("bot", data.reply || data.message || "I could not answer that yet.");
      return;
    }
    addBubble("bot", data.reply || "Here you go.");
    if (data.recommendation || data.kind === "answer" || data.kind === "memory") addPlanCard(data);
  } catch (error) {
    pending.remove();
    addBubble("bot", "I could not reach the server right now.");
  }
}

async function getPlan() {
  const payload = stateFromForm();
  addBubble("user", payload.description || "Get my gym plan");
  const pending = setPending();
  try {
    const { response, data } = await postJson("/recommend", { ...payload, state: { profile: profileStateFromForm() } });
    pending.remove();
    if (!response.ok || !data.ok) {
      addBubble("bot", data.reply || data.message || "Please fill a little more detail.");
      return;
    }
    addBubble("bot", data.reply || "Your plan is ready.");
    addPlanCard(data);
  } catch (error) {
    pending.remove();
    addBubble("bot", "I could not reach the server right now.");
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value;
  messageInput.value = "";
  sendChat(text);
});

planBtn.addEventListener("click", () => {
  getPlan();
});

document.querySelectorAll(".chip").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.fill || button.textContent.trim();
    messageInput.focus();
  });
});

messages.addEventListener("click", (event) => {
  const button = event.target.closest("[data-fill]");
  if (button) {
    messageInput.value = button.dataset.fill || "";
    messageInput.focus();
  }

  const feedbackButton = event.target.closest("[data-rating]");
  if (feedbackButton && lastAssistantPayload) {
    const rating = feedbackButton.dataset.rating;
    sendFeedback(rating, lastAssistantPayload);
    feedbackButton.textContent = rating === "up" ? "Thanks!" : "Got it";
    feedbackButton.disabled = true;
  }
});

addBubble("bot", "Ask me anything about gym training, meals, or body recomposition. You can also hit Get plan.");
loadOptions();
