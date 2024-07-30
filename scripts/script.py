import logging
import re
import gradio as gr
from openai import OpenAI
from modules import scripts
from modules.processing import StableDiffusionProcessingTxt2Img


def _get_effective_prompt(prompts: list[str], prompt: str) -> str:
    return prompts[0] if prompts else prompt


class JSPromptScript(scripts.Script):
    LLM_YOU = gr.State("a1")
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    llm_history_array = []

    def __init__(self) -> None:
        self.YOU_LLM = "A superstar on stage."
        super().__init__()

    def title(self):
        return "Dynamic Javascript Prompt"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def call_llm_pythonlib(self, _llm_system_prompt, _llm_ur_prompt, _llm_max_token, llm_recursive_use,
                           llm_keep_your_prompt_use):

        if llm_recursive_use and (self.llm_history_array.__len__() > 1):
            _llm_ur_prompt = (_llm_ur_prompt if llm_keep_your_prompt_use else "") + " " + self.llm_history_array[self.llm_history_array.__len__() - 1][0]
        completion = self.client.chat.completions.create(
            model="",
            messages=[
                {"role": "system", "content": _llm_system_prompt},
                {"role": "user", "content": _llm_ur_prompt}
            ],
            max_tokens=_llm_max_token,
            temperature=0.7,

        )
        print("[sd-webui-decadetw-auto-prompt-llm][Init-UI][completion]: " + str(completion) + "\n\n")
        self.llm_history_array.append([completion.choices[0].message.content, _llm_ur_prompt, _llm_system_prompt])
        if len(self.llm_history_array) > 3:
            self.llm_history_array.remove(self.llm_history_array[0])
        return completion.choices[0].message.content, self.llm_history_array

    def ui(self, is_img2img):
        # print("\n\n[][Init-UI][sd-webui-prompt-auto-llm]: " + str(is_img2img) + "\n\n")
        examples = [
            ["The Moon's orbit around Earth has"],
            ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
        ]

        with gr.Blocks():
            # gr.Markdown("Blocks")
            with gr.Accordion(open=True, label="Auto LLM Prompt v20240723"):
                with gr.Tab("Prompt/LLM-PythonLib"):
                    # with gr.Accordion(open=True, label="[Prompt]/[LLM-PythonLib]"):
                    gr.Markdown("* Generate forever mode \n"
                                "* Story board mode")
                    llm_is_enabled = gr.Checkbox(label="Enable LLM-Answer to SD-prompt", value=True)
                    llm_recursive_use = gr.Checkbox(label="Recursive-prompt. Use the prompt from last one", value=True)
                    llm_keep_your_prompt_use = gr.Checkbox(label="Keep Your prompt ahead each request", value=True)

                    with gr.Row():
                        with gr.Column():
                            llm_system_prompt = gr.Textbox(label="1. [LLM-System-Prompt]", lines=10,
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
                            llm_ur_prompt = gr.Textbox(label="2. [Your-Prompt]", lines=2, value="A superstar on stage.",
                                                       placeholder="A superstar on stage.")
                            llm_llm_answer = gr.Textbox(inputs=self.process, show_copy_button=True, interactive=True,
                                                        label="3. [LLM-Answer]", lines=6, placeholder="LLM says.")

                    llm_history = gr.Dataframe(
                        interactive=True,
                        label="History/StoryBoard",
                        headers=["llm_answer", "system_prompt", "ur_prompt"],
                        datatype=["str", "str", "str"],
                        row_count=3,
                        col_count=(3, "fixed"),
                    )

                    llm_max_token = gr.Slider(5, 500, value=50, step=5, label="Max LLM-answer length")
                    llm_button = gr.Button("Call LLM above")
                    llm_button.click(self.call_llm_pythonlib, inputs=[llm_system_prompt, llm_ur_prompt, llm_max_token],
                                     outputs=[llm_llm_answer, llm_history])

                with gr.Tab("Setup"):
                    llm_apiurl = gr.Textbox(label="0. [LLM-URL]", lines=1,
                                            placeholder="http://localhost:1234/v1")
                    llm_apikey = gr.Textbox(label="0. [LLM-API-Key]", lines=1,
                                            placeholder="lm-studio")
        return [llm_is_enabled, llm_recursive_use, llm_keep_your_prompt_use,
                llm_system_prompt, llm_ur_prompt, llm_llm_answer,
                llm_history,
                llm_max_token,
                llm_apiurl, llm_apikey]

    def process(self, p: StableDiffusionProcessingTxt2Img,
                llm_is_enabled, llm_recursive_use, llm_keep_your_prompt_use,
                llm_system_prompt, llm_ur_prompt, llm_llm_answer,
                llm_history,
                llm_max_token,
                llm_apiurl, llm_apikey):

        if llm_is_enabled:
            r = self.call_llm_pythonlib(llm_system_prompt, llm_ur_prompt, llm_max_token, llm_recursive_use,
                                        llm_keep_your_prompt_use)
            g_result = str(r[0])
            print("[][][]llm_llm_answer ", llm_llm_answer)

            for i in range(len(p.all_prompts)):
                p.all_prompts[i] = p.all_prompts[i] + ("," if (p.all_prompts[0].__len__() > 0) else "") + g_result

        return p.all_prompts[0]

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
