from asyncio.windows_events import NULL
import contextlib
import gradio as gr
import logging
from modules import scripts, script_callbacks, processing, shared

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from openai import OpenAI
import re

from modules.processing import StableDiffusionProcessingTxt2Img

def _get_effective_prompt(prompts: list[str], prompt: str) -> str:
    return prompts[0] if prompts else prompt

class JSPromptScript(scripts.Script):
    LLM_YOU = gr.State("a1")
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    llm_history_array = []
    defaultJS = """%% {
    resultX = await fetch('http://localhost:1234/v1/chat/completions', {
            method: 'POST',
            body: JSON.stringify({
                "messages": [{
                    "role": "system",
                    "content": "You are an AI prompt word playwright. Use the provided keywords to create a beautiful composition. You only need the prompt words, not your feelings. Customize the style, pose, lighting, scene, decoration, etc., and be as detailed as possible. "
                }, 
                {"role": "user", "content": "A superstar on sex."}],
                "temperature": 0.7,"max_tokens": 80,"stream": false
            }),
            headers: {'Content-Type': 'application/json'}
        }).then(res => {
            return res.json();
        }).then(result => {
            return result['choices'][0]['message']['content'];
        });
        return resultX;
    } %%"""

    def __init__(self) -> None:
        self.jr = NULL
        self.YOU_LLM = "A superstar on stage."
        # self.LLM_YOU = "AI"
        # self.LLM_YOU = gr.State()
        super().__init__()

    def title(self):
        return "Dynamic Javascript Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def call_llm_pythonlib(self, _llm_system_prompt, _llm_ur_prompt, _llm_max_token):
        completion = self.client.chat.completions.create(
            model="CHE-72/Qwen1.5-4B-Chat-Q2_K-GGUF/qwen1.5-4b-chat-q2_k.gguf",
            messages=[
                {"role": "system", "content": _llm_system_prompt},
                {"role": "user", "content": _llm_ur_prompt}
            ],
            max_tokens=_llm_max_token,
            temperature=0.7,
        )
        print("\n\n[][Init-UI][completion]: " + str(completion) + "\n\n")
        self.llm_history_array.append([completion.choices[0].message.content, _llm_ur_prompt, _llm_system_prompt] )
        if len(self.llm_history_array) > 3:
            self.llm_history_array.remove(self.llm_history_array[0])
        return completion.choices[0].message.content, self.llm_history_array

    def ui(self, is_img2img):
        print("\n\n[][Init-UI][sd-webui-prompt-auto-llm]: " + str(is_img2img) + "\n\n")
        examples = [
            ["The Moon's orbit around Earth has"],
            ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
        ]

        # def List_LLM_Models():
        #     completion = self.client.models.models()
        #     return completion

        def filter_records(records, gender):
            return records[records["gender"] == gender]

        with gr.Blocks():
            # gr.Markdown("Blocks")
            with gr.Accordion(open=True, label="Auto LLM Prompt v20240723"):
                with gr.Accordion(open=True, label="[Prompt]/[LLM-PythonLib]"):
                    gr.Markdown("Generate inside Main Prompt")
                    llm_is_enabled = gr.Checkbox(label="Enable LLM to prompt", value=True)

                    with gr.Row():
                        llm_system_prompt = gr.Textbox(label="0. [System-Prompt]", lines=10,
                                                       value="You are an AI prompt word "
                                                             "engineer. Use the provided "
                                                             "keywords to create a beautiful "
                                                             "composition. Only the prompt "
                                                             "words are needed, "
                                                             "not your feelings. Customize the "
                                                             "style, scene, decoration, etc., "
                                                             "and be as detailed as possible "
                                                             "without endings.",
                                                       placeholder="You are an AI prompt word "
                                                                   "engineer. Use the provided "
                                                                   "keywords to create a beautiful "
                                                                   "composition. Only the prompt "
                                                                   "words are needed, "
                                                                   "not your feelings. Customize the "
                                                                   "style, scene, decoration, etc., "
                                                                   "and be as detailed as possible "
                                                                   "without endings."
                                                       )
                        with gr.Column():
                            llm_ur_prompt = gr.Textbox(label="1. [YOU-Prompt]", lines=3, value="A superstar on stage.",
                                                       placeholder="A superstar on stage.")
                            llm_llm_answer = gr.Textbox(inputs=self.process, show_copy_button=True, interactive=True, label="2. [LLM-Answer]", lines=3, placeholder="LLM says.")

                    llm_history = gr.Dataframe(
                        interactive=True,
                        label="History",
                        headers=["llm_answer", "system_prompt", "ur_prompt"],
                        datatype=["str", "str", "str"],
                        row_count=3,
                        col_count=(3, "fixed"),
                    )

                    llm_max_token = gr.Slider(5, 500, value=50, step=5, label="Max length w/Generate token ")

                    llm_button = gr.Button("Call LLM above")
                    llm_button.click(self.call_llm_pythonlib, inputs=[llm_system_prompt, llm_ur_prompt, llm_max_token],
                                     outputs=[llm_llm_answer, llm_history])
                with gr.Accordion(open=False, label="[Prompt]/[LLM-JS]"):
                    gr.Markdown("Generate by JS / each generate")
                    js_is_enabled = gr.Checkbox(label=" Enable Dynamic Javascript", value=False)
                    js_howMany = gr.Slider(1, 20, value=2, step=1,
                                           label=" When Generate forever => ImagesCount/eachResult",
                                           info="repet use. (When call this JS get result; keep use the result on how many times)")
                    js_template = gr.Dropdown(
                        ['online'].append("online2"), value=['online'], multiselect=False,
                        label="List JS template(online)",
                        info="online URI: https://decade.tw/js_template"
                    )
                    with gr.Row():
                        js_prompt_js = gr.Textbox(label=" f[Prompt-JS]  %%{...}%%", lines=3,
                                                  placeholder=self.defaultJS)
                        js_result = gr.Textbox(label=" [Prompt-JS-Result]", lines=3, value="")
        return [js_is_enabled, llm_is_enabled,
                js_prompt_js, js_result,
                llm_system_prompt, llm_ur_prompt, llm_max_token,
                llm_llm_answer, llm_history]

    def process(self, p: StableDiffusionProcessingTxt2Img, js_is_enabled, llm_is_enabled,
                js_prompt_js, js_result,
                llm_system_prompt, llm_ur_prompt, llm_max_token,
                llm_llm_answer, llm_history):

        if llm_is_enabled:
            r = self.call_llm_pythonlib(llm_system_prompt, llm_ur_prompt, llm_max_token)
            g_result = str(r[0])
            print("[][][]llm_llm_answer ",llm_llm_answer)

            for i in range(len(p.all_prompts)):
                p.all_prompts[i] = p.all_prompts[i]+("," if (p.all_prompts[0].__len__() > 0) else "") + g_result

        if js_is_enabled:
            print("[][][]js_is_enabled-true")
            original_prompt = _get_effective_prompt(p.all_prompts, p.prompt)
            for i in range(len(p.all_prompts)):
                prompt = p.all_prompts[i]
                negative_prompt = p.all_negative_prompts[i]
                try:
                    if self.jr is NULL:
                        self.jr = JavascriptRunner()
                    if "%%" in prompt:
                        prompt = self.jr.execute_javascript_in_prompt(prompt)
                        # self.LLM_YOU = prompt
                        self.LLM_YOU = gr.State(prompt)
                        # self.LLM_YOU.change(prompt, inputs=[text2])

                    if "%%" in negative_prompt:
                        negative_prompt = self.jr.execute_javascript_in_prompt(negative_prompt)
                        self.LLM_YOU.append(" ,negative_llm_prompt=" + negative_prompt)

                    if self.jr is not NULL:
                        self.jr.resetContext()  # End shared context
                except Exception as e:
                    logging.exception(e)
                    prompt = str(e)
                    negative_prompt = str(e)

                p.all_prompts[i] = prompt
                p.all_negative_prompts[i] = negative_prompt
            p.prompt_for_display = original_prompt

        return p.all_prompts[0]

class JavascriptRunner:
    use_same_result_cout = 0
    lastPrompt = ""

    def __init__(self):
        options = Options()
        options.add_argument('--ignore-ssl-errors=yes')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)

    def __del__(self):
        self.driver.quit()

    def resetContext(self):
        # Navigate to a blank page after executing JS
        # This 'resets' things like variables so they don't hang around forever between executions
        self.driver.get('about:blank')

    def execute_javascript_in_prompt(self, prompt):
        JavascriptRunner.use_same_result_cout += 1
        print(">>>>>>>>>>>>>>>>>>" + str(JavascriptRunner.use_same_result_cout))

        sections = re.split('(%%.*?%%)', prompt, flags=re.DOTALL)
        for i, section in enumerate(sections):
            if section.startswith('%%') and section.endswith('%%'):
                # This is a JavaScript code section. Execute it.
                js_code = section[2:-2].strip()  # Remove the delimiters
                if JavascriptRunner.use_same_result_cout % 2 == 0:
                    result = JavascriptRunner.lastPrompt
                else:
                    result = self.driver.execute_script(js_code)
                    JavascriptRunner.lastPrompt = result
                # Replace the JavaScript code with its output in the sections list

                replacement = str(result)
                if replacement != 'None' and replacement.__len__() > 0:
                    sections[i] = replacement
                else:
                    sections[i] = ""

        # Join the sections back together into the final prompt
        # PAPA.LLM_YOU.value = ''.join(sections)
        # self.YOU_LLM = ''.join(sections)
        return ''.join(sections)

# with gr.Row():
#    js_neg_prompt_js = gr.Textbox(label="[Negative prompt-JS]", lines=3, value="{}")
#    js_neg_result = gr.Textbox(label="[Negative prompt-JS-Result]", lines=3, value="result")
#    # self.p.change(self.process, inputs=js_result, outputs=js_result)
# with gr.Row():
#     llm_models = gr.Dropdown(
#         ['noDetect'].append(List_LLM_Models), value=['noDetect'], multiselect=False,
#         label="List LLM "
#               "Models",
#         info="get models from local LLM. (:LM Studio)"
#     )
