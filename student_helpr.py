import streamlit as st
import base64
import re
import json
import os
import tempfile
import random
from openai import OpenAI

class StudentAIApp:
    def __init__(self):
        self.api_key = None
        self.client = None
        self.grade = None
        self.mode = None
        self.exam_type = None
        self.num_questions = 3   # ✅ default number of questions
        self.chapter_text = ""
        self.chapter_images = []
        # Learn mode preferences
        self.explain_style = "Summary + Step-by-step"
        self.custom_instructions = ""
        self.tts_enabled = False
        self.tts_voice = "alloy"
        # Branding
        self.app_title = "🎓 LearnSpark"
        self.app_tagline = "Understand better. Practice smarter."

    def render_header(self):
        mode_badge = self.mode or "Mode"
        st.markdown(
            f"""
            <style>
            .header-card {{
                background: linear-gradient(135deg, #f7f9ff 0%, #ffffff 100%);
                border: 1px solid #e8ecf7;
                padding: 18px 24px;
                border-radius: 16px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.04);
                margin-bottom: 8px;
            }}
            .header-title {{
                font-size: 32px;
                font-weight: 800;
                margin: 0;
                background: linear-gradient(90deg,#6c5ce7,#00b894);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .header-sub {{
                color: #555;
                font-size: 14px;
                margin-top: 6px;
            }}
            .badge {{
                display: inline-block;
                font-size: 12px;
                background: #eef3ff;
                color: #2d5bd1;
                padding: 4px 10px;
                border-radius: 999px;
                margin-left: 8px;
            }}
            </style>
            <div class="header-card">
              <h1 class="header-title">{self.app_title}</h1>
              <div class="header-sub">{self.app_tagline} </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Learn mode preferences
        self.explain_style = "Summary + Step-by-step"
        self.custom_instructions = ""

    # ---------- helpers for MCQ checking ----------
    def _strip_label(self, s: str) -> str:
        """
        Remove common MCQ prefixes like 'A)', 'B.', '1)', '2.' or bullets before comparing.
        """
        if s is None:
            return ""
        s = str(s)
        # remove bullet/markdown markers
        s = re.sub(r"^[\s>*•\-]+", "", s)
        # remove leading labels: A)  B.  (C)  1)  2. etc.
        s = re.sub(r"^\s*[\(\[]?([A-Za-z]|[0-9]+)[\)\].:\-]\s*", "", s)
        return s
    def _norm(self, s):
        if s is None:
            return None
        s = self._strip_label(s)
        s = re.sub(r"\s+", " ", str(s)).strip().lower()
        return s 
    def _resolve_mcq_answer(self, q):
        """
        Resolve the 'correct answer' into an expected index/text:
        Supports:
          - answer as the full option text
          - answer as a letter: 'A', 'b', ...
          - answer as a number: 0/1 based
          - answer as int
          - answer as dict with 'index' or 'text'
        Returns a dict:
            {
              'expected_idx': int or None,
              'expected_text': str or None,
              'mode': str,
              'raw_answer': original_answer
            }
        """
        options = q.get("options", []) or []
        answer = q.get("answer", None)

        options_norm = [self._norm(o) for o in options]
        result = {"expected_idx": None, "expected_text": None, "mode": "unknown", "raw_answer": answer}

        try:
            # If dict: possibly {"index": 2} or {"text": "Holi"}
            if isinstance(answer, dict):
                if "index" in answer:
                    idx = int(answer["index"])
                    if 0 <= idx < len(options):
                        result.update(expected_idx=idx, expected_text=options[idx], mode="dict_index0")
                        return result
                    if 1 <= idx <= len(options):
                        result.update(expected_idx=idx - 1, expected_text=options[idx - 1], mode="dict_index1")
                        return result
                if "text" in answer:
                    a_text = self._norm(answer["text"])
                    if a_text in options_norm:
                        idx = options_norm.index(a_text)
                        result.update(expected_idx=idx, expected_text=options[idx], mode="dict_text->option")
                        return result
                    result.update(expected_idx=None, expected_text=answer["text"], mode="dict_text_unmatched")
                    return result

            # If string: letter, number, or direct text
            if isinstance(answer, str):
                a = answer.strip()
                # letter like A/B/C/...
                if len(a) == 1 and a.upper().isalpha():
                    idx = ord(a.upper()) - ord("A")
                    if 0 <= idx < len(options):
                        result.update(expected_idx=idx, expected_text=options[idx], mode="letter")
                        return result
                # numeric string
                if re.fullmatch(r"\d+", a):
                    idx = int(a)
                    if 0 <= idx < len(options):  # 0-based
                        result.update(expected_idx=idx, expected_text=options[idx], mode="index0_str")
                        return result
                    if 1 <= idx <= len(options):  # 1-based
                        result.update(expected_idx=idx - 1, expected_text=options[idx - 1], mode="index1_str")
                        return result
                # direct text
                norm_a = self._norm(a)
                if norm_a in options_norm:
                    idx = options_norm.index(norm_a)
                    result.update(expected_idx=idx, expected_text=options[idx], mode="text->option")
                    return result
                # unmatched text (still return for display)
                result.update(expected_idx=None, expected_text=a, mode="text_unmatched")
                return result

            # If integer: index
            if isinstance(answer, int):
                if 0 <= answer < len(options):
                    result.update(expected_idx=answer, expected_text=options[answer], mode="index0_int")
                    return result
                if 1 <= answer <= len(options):
                    result.update(expected_idx=answer - 1, expected_text=options[answer - 1], mode="index1_int")
                    return result
        except Exception:
            pass

        return result
    def _mcq_check_and_explain(self, selected_value, q):
        """
        Step-by-step checker returning (is_correct, explanation_text, resolved_info).
        """
        options = q.get("options", []) or []
        options_norm = [self._norm(o) for o in options]
        sel_norm = self._norm(selected_value)

        resolved = self._resolve_mcq_answer(q)

        if resolved["expected_idx"] is not None:
            correct_norm = options_norm[resolved["expected_idx"]]
            correct_disp = options[resolved["expected_idx"]]
        else:
            correct_norm = self._norm(resolved["expected_text"])
            correct_disp = resolved["expected_text"]

        is_correct = (sel_norm == correct_norm)

        explanation = (
            "🔎 **MCQ Validation Details**\n\n"
            f"- Selected: `{selected_value}`\n"
            f"- Selected (normalized): `{sel_norm}`\n"
            f"- Resolution mode: `{resolved['mode']}`\n"
            f"- Correct (display): `{correct_disp}`\n"
            f"- Correct (normalized): `{correct_norm}`\n"
        )
        return is_correct, explanation, resolved
    def _mcq_is_correct(self, selected_value, q):
        """Wrapper around _mcq_check_and_explain so exam_mode works."""
        is_correct, _, _ = self._mcq_check_and_explain(selected_value, q)
        return is_correct
    # ---------- end helpers ----------

    def sidebar(self):
        st.sidebar.header("⚙️ Settings")
        self.api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")

        if not self.api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
            st.stop()

        self.client = OpenAI(api_key=self.api_key)

        # Grade & mode
        self.grade = st.sidebar.number_input("🎓 Enter your grade", min_value=1, max_value=12, step=1, value=3)
        self.mode = st.sidebar.radio("Choose Mode:", ["📖 Learn", "📝 Exam"], index=0)

        if self.mode == "📝 Exam":
            exam_type = st.sidebar.radio("Choose Exam Type:", ["Theory", "MCQ"])

            # ✅ Number of questions input
            self.num_questions = st.sidebar.number_input(
                "How many questions?",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )

            # Reset exam questions only when exam type actually changes across reruns
            prev_exam_type = st.session_state.get("exam_type")
            if prev_exam_type is not None and prev_exam_type != exam_type:
                st.session_state.pop("exam_questions", None)
            st.session_state.exam_type = exam_type
            self.exam_type = exam_type

        if self.mode == "📖 Learn":
            st.sidebar.markdown("---")
            self.explain_style = st.sidebar.selectbox(
                "Explanation style",
                [
                    "Summary + Step-by-step",
                    "Bullet points",
                    "Examples first",
                    "Analogies for kids",
                    "Detailed with definitions",
                ],
                index=0,
            )
            self.custom_instructions = st.sidebar.text_area(
                "Optional extra instructions (tone, language, focus topics)",
                placeholder="e.g., Use simple words and include 2 real-life examples",
                height=80,
            )
            self.tts_enabled = st.sidebar.checkbox("🔊 Read answers aloud")
            self.tts_voice = st.sidebar.selectbox(
                "Voice",
                ["alloy", "verse", "amber", "aria"],
                index=0,
            )

    def get_lesson_input(self):
        st.subheader("📚 Provide Chapter Material")
        chapter_name = st.text_input("Enter chapter name (optional)")
        pasted_text = st.text_area("Or paste text from the chapter")
        uploaded_images = st.file_uploader(
            "Or upload multiple chapter images (jpg/png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        # Fingerprint lesson input to detect changes across reruns
        new_text = chapter_name or pasted_text
        current_images = uploaded_images if not new_text else []
        new_fp = {
            "text": new_text or "",
            "num_images": len(current_images),
            "image_names": [img.name for img in current_images],
        }
        prev_fp = st.session_state.get("lesson_fp")
        if prev_fp != new_fp:
            st.session_state.pop("exam_questions", None)
            # Refresh persisted image bytes and mime types
            image_bytes_list = []
            image_mime_list = []
            for img in current_images:
                # Read bytes once and store; avoid reusing consumed buffers later
                try:
                    img.seek(0)
                except Exception:
                    pass
                data = img.read()
                if not data:
                    continue
                image_bytes_list.append(data)
                mime = getattr(img, "type", None)
                if not mime:
                    name = getattr(img, "name", "")
                    if name.lower().endswith((".jpg", ".jpeg")):
                        mime = "image/jpeg"
                    elif name.lower().endswith((".png",)):
                        mime = "image/png"
                    else:
                        mime = "application/octet-stream"
                image_mime_list.append(mime)

            st.session_state.uploaded_images_bytes = image_bytes_list
            st.session_state.uploaded_images_mime_types = image_mime_list
            st.session_state.lesson_fp = new_fp

        self.chapter_text = new_text
        self.chapter_images = current_images

    def clean_json(self, text):
        """Extract JSON safely using regex"""
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return None
        except Exception:
            return None 

    def learn_mode(self):
        st.subheader("📖 Learn Mode")

        if not (self.chapter_text or self.chapter_images):
            st.info("Please provide chapter material first.")
            return

        # --- Collect material (text + images) ---
        user_content = []
        if self.chapter_text:
            user_content.append({"type": "text", "text": f"Explain for Grade {self.grade}: {self.chapter_text}"})
        # Prefer persisted image bytes (set in get_lesson_input) to avoid empty reads
        image_bytes_list = st.session_state.get("uploaded_images_bytes", [])
        image_mime_list = st.session_state.get("uploaded_images_mime_types", [])
        if not image_bytes_list and self.chapter_images:
            # Fallback: read current uploaded files and persist for future reruns
            image_bytes_list = []
            image_mime_list = []
            for img in self.chapter_images:
                try:
                    img.seek(0)
                except Exception:
                    pass
                data = img.read()
                if not data:
                    continue
                image_bytes_list.append(data)
                mime = getattr(img, "type", None)
                if not mime:
                    name = getattr(img, "name", "")
                    if name.lower().endswith((".jpg", ".jpeg")):
                        mime = "image/jpeg"
                    elif name.lower().endswith((".png",)):
                        mime = "image/png"
                    else:
                        mime = "application/octet-stream"
                image_mime_list.append(mime)
            st.session_state.uploaded_images_bytes = image_bytes_list
            st.session_state.uploaded_images_mime_types = image_mime_list

        for idx, img_bytes in enumerate(image_bytes_list):
            if not img_bytes:
                continue
            mime = image_mime_list[idx] if idx < len(image_mime_list) else "image/png"
            if mime == "application/octet-stream":
                # Skip unknown types to avoid invalid data URLs
                continue
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_img}"}
            })

        # --- Lesson Summary ---
        if not user_content:
            st.info("Please provide chapter material first.")
            return
        style_map = {
            "Summary + Step-by-step": "Provide a short summary followed by a step-by-step explanation.",
            "Bullet points": "Explain using concise bullet points only.",
            "Examples first": "Start with 2-3 concrete examples, then explain the concept.",
            "Analogies for kids": "Use simple analogies appropriate for a Grade student.",
            "Detailed with definitions": "Give clear definitions, then detailed explanation with subheadings.",
        }
        style_instruction = style_map.get(self.explain_style, "Provide a short summary followed by a step-by-step explanation.")
        extra = (self.custom_instructions or "").strip()

        system_message = (
            f"You are a helpful teacher for Grade {self.grade}. "
            f"Explanation style: {self.explain_style}. {style_instruction} "
            + (f"Extra instructions: {extra}" if extra else "")
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
        )

        st.markdown("### 📝 Lesson Summary")
        summary_text = response.choices[0].message.content
        st.write(summary_text)

        # Optional TTS for summary
        if self.tts_enabled and summary_text:
            try:
                speech = self.client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice=self.tts_voice,
                    input=summary_text,
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(speech.content)
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                os.unlink(tmp_path)
            except Exception as e:
                st.info(f"Speech unavailable: {e}")

        # --- Chat Section ---
        st.markdown("### 💬 Chat About This Chapter")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if query := st.chat_input("Ask a question about the chapter..."):
            # Save user question
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Build messages: system + chapter material + chat history
            base_context = [
                {"role": "system", "content": (
                    "You are a helpful tutor. Always use the provided chapter text/images as the main reference. "
                    "If an answer is not present in the material, reply: 'This information is not available in the chapter.' "
                    f"Follow this explanation style: {self.explain_style}. {style_instruction} "
                    + (f"Extra instructions: {extra}" if extra else "")
                )},
                {"role": "user", "content": user_content}  # inject material every time
            ]
            full_context = base_context + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_resp = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=full_context
                        )
                        answer = chat_resp.choices[0].message.content
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"

                    st.markdown(answer)

                    # Optional TTS for chat answer
                    if self.tts_enabled and answer:
                        try:
                            speech = self.client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice=self.tts_voice,
                                input=answer,
                            )
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                                tmp.write(speech.content)
                                tmp_path = tmp.name
                            with open(tmp_path, "rb") as f:
                                st.audio(f.read(), format="audio/mp3")
                            os.unlink(tmp_path)
                        except Exception as e:
                            st.info(f"Speech unavailable: {e}")

            # Save AI reply
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Optional clear chat (reset everything)
        if st.button("🗑️ Clear Chat"):
            for _k in list(st.session_state.keys()):
                del st.session_state[_k]
            st.rerun()


    def exam_mode(self):
        st.subheader("📝 Exam Mode")

        if not (self.chapter_text or self.chapter_images):
            st.info("Please provide chapter material first.")
            return

        # Ensure persisted image bytes exist (they are set in get_lesson_input when inputs change)
        if "uploaded_images_bytes" not in st.session_state:
            st.session_state.uploaded_images_bytes = []

        def generate_questions():
            user_content = []
            if self.chapter_text:
                user_content.append({
                    "type": "text",
                    "text": f"Generate {self.num_questions} {self.exam_type} exam-style questions for Grade {self.grade}: {self.chapter_text}"
                })
            image_bytes_list = st.session_state.get("uploaded_images_bytes", [])
            image_mime_list = st.session_state.get("uploaded_images_mime_types", [])
            for idx, img_bytes in enumerate(image_bytes_list):
                if not img_bytes:
                    continue
                mime = image_mime_list[idx] if idx < len(image_mime_list) else "image/png"
                if mime == "application/octet-stream":
                    continue
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_img}"}
                })
            # Guard: Do not call model if no material
            if not user_content:
                return None
            if self.exam_type == "Theory":
                system_message = f"""
                You are an exam question generator. 
                Generate exactly {self.num_questions} **theory** questions only.
                Return valid JSON in this format:
                [
                {{"id": 1, "type": "theory", "question": "Why do people celebrate Diwali in India?"}}
                ]
                """ 
            elif self.exam_type == "MCQ":
                system_message = f"""
                You are an exam question generator. 
                Generate exactly {self.num_questions} **multiple-choice (mcq)** questions only.
                Each MCQ must have 4 options and exactly 1 correct answer.
                Return valid JSON in this format:
                [
                {{"id": 1, "type": "mcq", "question": "Which festival is known as the festival of lights?", 
                    "options": ["Diwali", "Holi", "Eid", "Christmas"], "answer": "Diwali"}}
                ]
                """ 
           
           
            exam_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ]
            )
            return self.clean_json(exam_response.choices[0].message.content)

        # --- First time generation ---
        if "exam_questions" not in st.session_state:
            parsed = generate_questions()
            if parsed:
                st.session_state.exam_questions = parsed
            else:
                st.error("❌ Failed to parse exam questions. Please retry.")
                return

        # --- Display questions ---
        st.markdown("### 📌 Exam Questions")
        for q in st.session_state.exam_questions:
            with st.container():
                st.markdown(f"**Q{q['id']}: {q['question']}**")

                if q["type"] == "theory":
                    student_answer = st.text_area(
                        f"✍️ Your Answer for Q{q['id']}",
                        key=f"theory_answer_{q['id']}"  # ✅ unique key
                    )
                    if st.button(f"✅ Check Answer Q{q['id']}", key=f"check_theory_{q['id']}"):
                        eval_response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": f"Evaluate answers for Grade {self.grade}. Give score (0–100) and simple feedback."},
                                {"role": "user", "content": f"Question: {q['question']}\nStudent Answer: {student_answer}"}
                            ]
                        )
                        feedback = eval_response.choices[0].message.content
                        st.markdown("### 📊 Feedback")
                        st.success(feedback)

                elif q["type"] == "mcq":
                    selected_key = f"answer_{q['id']}"
                    student_answer = st.radio(
                        label="",  # collapse duplicate question text
                        options=q["options"],
                        key=selected_key,
                        label_visibility="collapsed",
                    )
                    print('keys',selected_key)
                    if st.button(f"✅ Check Answer Q{q['id']}", key=f"check_mcq_{q['id']}"):
                        chosen = st.session_state.get(selected_key, None)

                        if chosen is None:
                            st.warning("⚠️ Please select an answer first.")
                        else:
                            is_correct, explanation, resolved = self._mcq_check_and_explain(chosen, q)
                            if is_correct:
                                st.success("✅ Correct!")
                            else:
                                correct_display = resolved.get("expected_text")
                                st.error(f"❌ Incorrect! Correct answer: {correct_display}")
                            with st.expander("See validation details"):
                                st.markdown(explanation)

        # --- Generate more button ---
        if st.button("➕ Generate More Questions"):
            new_questions = generate_questions()
            if new_questions:
                # Increment IDs to avoid duplicates
                max_id = max(q["id"] for q in st.session_state.exam_questions)
                for i, q in enumerate(new_questions, start=1):
                    q["id"] = max_id + i
                st.session_state.exam_questions.extend(new_questions)
                st.rerun()
            else:
                st.error("❌ Failed to generate additional questions. Please try again.")

    def run(self):
        st.set_page_config(page_title="LearnSpark", page_icon="🎓", layout="wide")
        self.sidebar()
        self.render_header()

        main_tabs = st.tabs(["📘 Study & Exam", "➗ Math Practice"])

        with main_tabs[0]:
            self.get_lesson_input()
            if self.mode == "📖 Learn":
                self.learn_mode()
            elif self.mode == "📝 Exam":
                self.exam_mode()

        with main_tabs[1]:
            st.subheader("➗ Math Practice")

            st.markdown("Select operations, difficulty, and generate practice MCQs. No internet or AI needed.")

            col_a, col_b = st.columns([2, 1])
            with col_a:
                ops = st.multiselect(
                    "Operations",
                    ["Addition (+)", "Subtraction (-)", "Multiplication (×)", "Division (÷)"],
                    default=["Addition (+)"]
                )
            with col_b:
                difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=0)

            combine_ops = st.checkbox("Combine selected operations in one expression", value=False)
            math_num_questions = st.number_input("How many questions?", min_value=1, max_value=20, value=5, step=1)
            style_hint = st.text_area(
                "Optional example/style (how you want the question to look)",
                placeholder="e.g., Make it a word problem about apples and shopping",
                height=60,
            )
            generator_mode = st.radio("Question source", ["Local (offline)", "LLM (OpenAI)"], index=0)

            def pick_operand(level: str):
                if level == "Easy":
                    return random.randint(1, 10)
                if level == "Medium":
                    return random.randint(5, 50)
                return random.randint(10, 100)

            def ensure_divisible(a: int, b: int) -> tuple:
                if b == 0:
                    b = 1
                a = (a // b) * b
                if a == 0:
                    a = b
                return a, b

            def compute(a: int, op: str, b: int) -> int:
                if op == "+":
                    return a + b
                if op == "-":
                    return a - b
                if op == "×":
                    return a * b
                if op == "÷":
                    if b == 0:
                        return a
                    return a // b
                return 0

            def gen_single(ops_labels, level, combine, hint) -> dict:
                label_to_op = {
                    "Addition (+)": "+",
                    "Subtraction (-)": "-",
                    "Multiplication (×)": "×",
                    "Division (÷)": "÷",
                }
                available_ops = [label_to_op[o] for o in ops_labels] or ["+"]

                if not combine or len(available_ops) == 1:
                    op = random.choice(available_ops)
                    a = pick_operand(level)
                    b = pick_operand(level)
                    if op == "÷":
                        a, b = ensure_divisible(a, b)
                    correct = compute(a, op, b)
                    q_text = f"Compute: {a} {op} {b}"
                else:
                    op1 = random.choice(available_ops)
                    op2 = random.choice(available_ops)
                    a = pick_operand(level)
                    b = pick_operand(level)
                    c = pick_operand(level)
                    if op1 == "÷":
                        a, b = ensure_divisible(a, b)
                    if op2 == "÷":
                        ab = compute(a, op1, b)
                        ab, c = ensure_divisible(ab, c)
                    ab = compute(a, op1, b)
                    correct = compute(ab, op2, c)
                    q_text = f"Compute: ({a} {op1} {b}) {op2} {c}"

                if hint:
                    q_text = f"{hint.strip()}\n{q_text}"

                # Generate distractors
                distractors = set()
                tweaks = [-2, -1, 1, 2, 3, -3]
                # Wrong precedence for combined expressions (if applicable)
                if combine and len(available_ops) > 1:
                    # Attempt precedence-based wrong answer: a op1 (b op2 c)
                    try:
                        if op2 == "÷":
                            bc_a = b
                            bc_b = c if c != 0 else 1
                            bc_a, bc_b = ensure_divisible(bc_a, bc_b)
                            bc = compute(bc_a, op2, bc_b)
                        else:
                            bc = compute(b, op2, c)
                        wrong_prec = compute(a, op1, bc)
                        if wrong_prec != correct:
                            distractors.add(wrong_prec)
                    except Exception:
                        pass
                while len(distractors) < 3:
                    off = random.choice(tweaks)
                    cand = correct + off
                    if cand != correct:
                        distractors.add(cand)
                options = [str(correct)] + [str(x) for x in list(distractors)[:3]]
                random.shuffle(options)
                return {
                    "id": 0,
                    "type": "mcq",
                    "question": q_text,
                    "options": options,
                    "answer": str(correct),
                }

            def _normalize_ops_for_display(expr: str) -> str:
                expr = expr.replace('*', '×').replace('x', '×').replace('X', '×')
                expr = expr.replace('/', '÷').replace('÷', '÷')
                return expr

            def _normalize_ops_for_eval(expr: str) -> str:
                expr = expr.replace('×', '*').replace('x', '*').replace('X', '*')
                expr = expr.replace('÷', '/')
                return expr

            def parse_hint_arithmetic(hint_text: str):
                if not hint_text:
                    return None
                # Extract a candidate arithmetic substring
                candidates = re.findall(r"[\d\s\+\-\*/xX÷×\(\)]+", hint_text)
                candidates = [c.strip() for c in candidates if c and any(op in c for op in ['+', '-', '*', 'x', 'X', '÷', '×', '/']) and re.search(r"\d", c)]
                if not candidates:
                    return None
                # Choose the longest plausible candidate
                expr = max(candidates, key=len)
                expr_eval = _normalize_ops_for_eval(expr)
                # Ensure only safe characters
                if not re.fullmatch(r"[0-9\s\+\-\*/\(\)]+", expr_eval):
                    return None
                try:
                    result = eval(expr_eval, {"__builtins__": None}, {})
                except Exception:
                    return None
                if isinstance(result, float):
                    # Only accept if integral
                    if abs(result - round(result)) < 1e-9:
                        result = int(round(result))
                    else:
                        return None
                if not isinstance(result, int):
                    return None
                expr_display = _normalize_ops_for_display(expr)
                return {"expr_display": expr_display, "result": result}

            def gen_from_hint(parsed):
                expr_display = parsed["expr_display"]
                correct = parsed["result"]
                q_text = f"Compute: {expr_display}"
                distractors = set()
                tweaks = [-2, -1, 1, 2, 3, -3]
                while len(distractors) < 3:
                    off = random.choice(tweaks)
                    cand = correct + off
                    if cand != correct:
                        distractors.add(cand)
                options = [str(correct)] + [str(x) for x in list(distractors)[:3]]
                random.shuffle(options)
                return {
                    "id": 0,
                    "type": "mcq",
                    "question": q_text,
                    "options": options,
                    "answer": str(correct),
                }

            def generate_with_llm(ops_labels, level, combine, count, hint):
                if not ops_labels:
                    return None
                # Map difficulty to constraints
                if level == "Easy":
                    range_text = "Use small integers, mostly 1-10."
                elif level == "Medium":
                    range_text = "Use integers roughly 5-50."
                else:
                    range_text = "Use integers roughly 10-100."

                ops_text = ", ".join(ops_labels)
                combine_text = (
                    "Some questions should combine two selected operations in a single expression."
                    if combine else
                    "Use only a single operation per question."
                )
                style_text = f"Style hint: {hint.strip()}" if hint and hint.strip() else ""

                system_message = (
                    "You are a math question generator for school students. "
                    f"Target grade: {self.grade}. Generate clear, unambiguous arithmetic MCQs."
                )
                user_prompt = f"""
Generate exactly {count} multiple-choice math questions.
Operations allowed: {ops_text}.
{combine_text}
Difficulty guideline: {range_text}
{style_text}

Rules:
- Each question must be arithmetic with an integer correct answer.
- Provide exactly 4 options; ensure only ONE option is correct, others are plausible.
- Keep numbers manageable; avoid fractions and decimals; if division appears, make it exact.
- Return ONLY valid JSON in this exact shape:
[
  {{"id": 1, "type": "mcq", "question": "...", "options": ["...","...","...","..."], "answer": "..."}}
]
                """

                try:
                    resp = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    parsed = self.clean_json(resp.choices[0].message.content)
                    # basic sanity check
                    if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                        # Normalize ids sequentially
                        for i, qd in enumerate(parsed, start=1):
                            qd["id"] = i
                            qd["type"] = "mcq"
                        return parsed
                    return None
                except Exception as e:
                    st.error(f"LLM error: {e}")
                    return None

            if st.button("Generate Math Questions"):
                if not ops:
                    st.warning("Please select at least one operation.")
                else:
                    if generator_mode == "Local (offline)":
                        items = []
                        parsed_hint = parse_hint_arithmetic(style_hint)
                        for i in range(math_num_questions):
                            if i == 0 and parsed_hint is not None:
                                qd = gen_from_hint(parsed_hint)
                            else:
                                qd = gen_single(ops, difficulty, combine_ops, style_hint)
                            qd["id"] = i + 1
                            items.append(qd)
                        st.session_state.math_questions = items
                    else:
                        llm_items = generate_with_llm(ops, difficulty, combine_ops, math_num_questions, style_hint)
                        if llm_items:
                            st.session_state.math_questions = llm_items
                        else:
                            st.error("❌ Failed to generate questions with LLM. Try adjusting options or use Local mode.")

            # Display math questions
            if "math_questions" in st.session_state:
                st.markdown("### 📌 Math Questions")
                for q in st.session_state.math_questions:
                    with st.container():
                        st.markdown(f"**Q{q['id']}: {q['question']}**")
                        selected_key = f"math_answer_{q['id']}"
                        st.radio(
                            label="",
                            options=q.get("options", []),
                            key=selected_key,
                            label_visibility="collapsed",
                        )
                        if st.button(f"✅ Check Answer Q{q['id']}", key=f"check_math_{q['id']}"):
                            chosen = st.session_state.get(selected_key, None)
                            if chosen is None:
                                st.warning("⚠️ Please select an answer first.")
                            else:
                                is_correct, explanation, resolved = self._mcq_check_and_explain(chosen, q)
                                if is_correct:
                                    st.success("✅ Correct!")
                                else:
                                    st.error(f"❌ Incorrect! Correct answer: {resolved.get('expected_text')}")
                                with st.expander("See validation details"):
                                    st.markdown(explanation)


if __name__ == "__main__":
    app = StudentAIApp()
    app.run()
