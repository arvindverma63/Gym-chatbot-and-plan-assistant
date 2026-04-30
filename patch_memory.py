import re

with open('assistant_core.py', 'r', encoding='utf-8') as f:
    code = f.read()

init_old = """        self.memory_path = self.base_dir / "memory.json"
        self.feedback_path = self.base_dir / "feedback.json"
        self._mtimes: dict[str, float] = {}
        self.advisor: GymAdvisor | None = None
        self.exercises: list[dict] = []
        self.visual_exercises: list[dict] = []
        self.nutrition: list[dict] = []
        self.knowledge: list[dict] = []
        self.knowledge_index: dict[str, list[int]] = {}
        self.knowledge_by_intent: dict[str, list[int]] = {}
        self.memory: dict[str, Any] = {"profile": {}, "interactions": []}
        self.feedback: list[dict] = []
        self._memory_changed = False
        self._feedback_changed = False
        self._load_memory()
        self._load_feedback()
        self.refresh(force=True)"""

init_new = """        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.feedback_path = self.base_dir / "feedback.json"
        self._mtimes: dict[str, float] = {}
        self.advisor: GymAdvisor | None = None
        self.exercises: list[dict] = []
        self.visual_exercises: list[dict] = []
        self.nutrition: list[dict] = []
        self.knowledge: list[dict] = []
        self.knowledge_index: dict[str, list[int]] = {}
        self.knowledge_by_intent: dict[str, list[int]] = {}
        self.feedback: list[dict] = []
        self._feedback_changed = False
        self._load_feedback()
        self.refresh(force=True)"""

code = code.replace(init_old, init_new)
code = code.replace("self._memory_changed = False\n        self._feedback_changed = False", "self._feedback_changed = False")

mem_old = """    def _load_memory(self) -> None:
        if not self.memory_path.exists():
            self.memory = {"profile": {}, "interactions": []}
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self.memory = {
                    "profile": data.get("profile", {}) if isinstance(data.get("profile"), dict) else {},
                    "interactions": data.get("interactions", []) if isinstance(data.get("interactions"), list) else [],
                }
        except Exception:
            self.memory = {"profile": {}, "interactions": []}

    def _save_memory(self) -> None:
        if not self._memory_changed:
            return
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, indent=2, ensure_ascii=False)
        self._memory_changed = False"""

mem_new = """    def _load_memory(self, session_id: str) -> dict:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "_-") or "default"
        path = self.sessions_dir / f"{safe_id}.json"
        if not path.exists():
            return {"profile": {}, "interactions": []}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return {
                    "profile": data.get("profile", {}) if isinstance(data.get("profile"), dict) else {},
                    "interactions": data.get("interactions", []) if isinstance(data.get("interactions"), list) else [],
                }
        except Exception:
            pass
        return {"profile": {}, "interactions": []}

    def _save_memory(self, session_id: str, memory: dict) -> None:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "_-") or "default"
        path = self.sessions_dir / f"{safe_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, indent=2, ensure_ascii=False)"""

code = code.replace(mem_old, mem_new)

code = code.replace('def recommend(self, payload: dict) -> dict:',
'def recommend(self, payload: dict) -> dict:\n        session_id = payload.get("session_id", "default")\n        memory = self._load_memory(session_id)')

code = code.replace('def _memory_match(self, message: str) -> Optional[dict]:', 'def _memory_match(self, message: str, memory: dict) -> Optional[dict]:')
code = code.replace('self.memory.get("interactions", []) if isinstance(self.memory.get("interactions"), list) else []', 'memory.get("interactions", []) if isinstance(memory.get("interactions"), list) else []')

code = code.replace('def chat(self, payload: dict) -> dict:',
'def chat(self, payload: dict) -> dict:\n        session_id = payload.get("session_id", "default")\n        memory = self._load_memory(session_id)')

code = code.replace('self._memory_match(message)', 'self._memory_match(message, memory)')

code = code.replace('self.memory[', 'memory[')
code = code.replace('self.memory.get', 'memory.get')
code = code.replace('self.memory.setdefault', 'memory.setdefault')
code = code.replace('self._save_memory()', 'self._save_memory(session_id, memory)')
code = code.replace('        self._memory_changed = True\n', '')

with open('assistant_core.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched successfully")
