import base64
import json
import logging
import os
import enum
# import pprint
import random
import subprocess
from io import BytesIO
import re
import gradio as gr
import requests
from modules import scripts
from modules.api.models import value
from modules.hashes import cache
from modules.processing import StableDiffusionProcessingTxt2Img
# from modules.script_callbacks import on_ui_tabs
from modules.shared import opts

log = logging.getLogger("[auto-llm]")
# log.setLevel(logging.INFO)
# Logging
log_file = os.path.join(scripts.basedir(), "auto-llm.log")

random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5'  # ⇅
restore_progress_symbol = '\U0001F300'  # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐


# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

class Enum_Api_MODEL_ReturnType(enum.Enum):
    l31 = 'llama3.1'
    gpro = 'gemini-pro'
    gpro_v = 'gemini-1.5-flash'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class Enum_Api_URL_ReturnType(enum.Enum):
    LMStudio = 'http://localhost:1234/v1'
    ollama = 'http://localhost:11434/v1'
    gemini = 'https://generativelanguage.googleapis.com/v1'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class EnumCmdReturnType(enum.Enum):
    JUST_CALL = 'just-call'
    LLM_USER_PROMPT = 'LLM-USER-PROMPT'
    LLM_VISION_IMG_PATH = 'LLM-VISION-IMG_PATH'

    @classmethod
    def values(cls):
        return [e.value for e in cls]


def xprint(obj):
    for attr in dir(obj):
        if not attr.startswith("__"):
            print(attr + "==>", getattr(obj, attr))


def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def _get_effective_prompt(prompts: list[str], prompt: str) -> str:
    return prompts[0] if prompts else prompt


def read_from_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        return json.load(file)


def write_to_file(filename, current_ui_settings):
    with open(filename, "w", encoding="utf8") as file:
        json.dump(current_ui_settings, file, indent=4, ensure_ascii=False)


def community_export_to_text(*args, **kwargs):
    dictx = (dict(zip(all_var_key, args)))
    write_to_file('Auto-LLM-settings.json', dictx)
    return json.dumps(dictx, indent=4)


def community_import_from_text(*args, **kwargs):
    try:
        if len(str(args[0])) <= 0:
            log.warning("[?][Auto-LLM][lOADING]Auto-LLM-settings.json")
            jo = read_from_file('Auto-LLM-settings.json')
            log.warning("[o][Auto-LLM][lOADED]Auto-LLM-settings.json")
        else:
            jo = json.loads(args[0])
        import_data = []
        for ele in all_var_key:
            import_data.append(jo[ele])
        log.warning("[O][Auto-LLM][Import-OK]")
        return import_data  #.append(json.dumps(jo, indent=4))
    except Exception as e:
        log.warning("[X][Auto-LLM][Import-Fail]")


def getReqJson(llm_apiurl, llm_apikey, llm_api_model_name, llm_text_system_prompt, llm_text_ur_prompt,
               llm_text_tempture, llm_top_k_text, llm_top_p_text, llm_text_max_token, base64_image,
               llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_max_token_eye, llm_text_tempture_eye,
               llm_top_p_vision):
    if 'google' in llm_apiurl:
        if len(base64_image) > 0:
            j = {
                'contents': [
                    {"role": "user",
                     "parts": [{
                         "text": f"{llm_text_ur_prompt}",
                     }, {
                         "inline_data": {
                             "mime_type": "image/jpeg",
                             "data": base64_image
                         }
                     }
                     ]}
                ],
                "generationConfig": {
                    "temperature": llm_text_tempture,
                    "topK": llm_top_k_text,
                    "topP": llm_top_p_text,
                    "maxOutputTokens": llm_text_max_token
                    #"responseMimeType": "text/plain"
                }
            }
        else:
            j = {
                'contents': [
                    {"role": "user",
                     "parts": [{
                         "text": f"{llm_text_ur_prompt}",
                     }
                     ]}
                ],
                "generationConfig": {
                    "temperature": llm_text_tempture,
                    "topK": llm_top_k_text,
                    "topP": llm_top_p_text,
                    "maxOutputTokens": llm_text_max_token
                    # "responseMimeType": "text/plain"
                }
            }
    else:
        if len(base64_image) > 0:
            j = {
                'model': f"{llm_api_model_name}",
                'messages': [
                    {
                        "role": "system",
                        "content": f"{llm_text_system_prompt_eye}",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{llm_text_ur_prompt_eye}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],

                'max_tokens': llm_text_max_token_eye,
                'temperature': llm_text_tempture_eye,
                'top_p': llm_top_p_vision,
                'stream': False,
            }
        else:
            j = {
                'model': f'{llm_api_model_name}',
                'messages': [
                    {'role': 'system', 'content': f'{llm_text_system_prompt}'},
                    {'role': 'user', 'content': f'{llm_text_ur_prompt}'}
                ],
                'max_tokens': llm_text_max_token,
                'temperature': llm_text_tempture,
                'top_p': llm_top_p_text,
                "top_k": llm_top_k_text,
                'stream': False,
            }
    # log.warning(f"[][AutoLLM][getReq][Json]{j}")

    return j


class AutoLLM(scripts.Script):
    # client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    llm_history_array = []
    llm_history_array_eye = []
    webpage_walker_array = []
    llm_sys_vision_template = (
        "You are an AI prompt word engineer. Use the provided image to create a beautiful composition. Only the prompt words are needed, not your feelings. Customize the style, scene, decoration, etc., and be as detailed as possible without endings.")

    llm_sys_text_template = (
        "You are an AI prompt word engineer. Use the provided keywords to create a beautiful composition. Only the prompt words are needed, not your feelings. Customize the style, scene, decoration, etc., and be as detailed as possible without endings.")
    llm_sys_translate_template = ("You are a professional and literary Taiwanese translation expert."
                                  "Please follow the following rules to translate into Taiwanese Traditional Chinese:)"
                                  "\n- Only the translated text is returned without any explanation."
                                  "\n- Language: Use Traditional Chinese and Taiwanese idioms for translation, do not use Simplified Chinese and Chinese idioms."
                                  "\n- Style: In line with Taiwanese writing habits, it is smooth and easy to read, and strives to be literary and meaningful."
                                  "\n- Nouns: Translate movie titles, book titles, authors, and artist names using Taiwanese common translation methods. Noun translations within the same article must be consistent."
                                  "\n- Format: All punctuation marks must be full-width, with spaces between Chinese and English."
                                  "\n- Each sentence should not exceed 30 words."
                                  "\n- Avoid inverted sentences.")

    def __init__(self) -> None:
        self.llm_llm_answer = None
        self.llm_ans_state = None
        self.YOU_LLM = "A superstar on stage."
        super().__init__()

    def title(self):
        return "Auto LLM"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # def check_api_uri(self, llm_apiurl, llm_apikey):
    #     if self.client.base_url != llm_apiurl or self.client.api_key != llm_apikey:
    #         self.client = OpenAI(base_url=llm_apiurl, api_key=llm_apikey)

    def CMG_getter(self, CivitaiMetaGrabber_url: str, CivitaiMetaGrabber_target_tag, CivitaiMetaGrabber_page,
                   CivitaiMetaGrabber_limit,
                   CivitaiMetaGrabber_model_id, CivitaiMetaGrabber_model_version_id):
        #https://github.com/civitai/civitai/wiki/REST-API-Reference/dff336bf9450cb11e80fb5a42327221ce3f09b45#get-apiv1images
        regex = r"^(?:urn:)?(?:air:)?(?:([a-zA-Z0-9_\-\/]+):)?(?:([a-zA-Z0-9_\-\/]+):)?([a-zA-Z0-9_\-\/]+):([a-zA-Z0-9_\-\/]+)(?:@([a-zA-Z0-9_\-]+))?(?:\.([a-zA-Z0-9_\-]+))?$"
        headers = {'user-agent': 'Mozilla/5.0'}
        result = []
        if str(CivitaiMetaGrabber_url).startswith("urn"):
            matches = re.match(regex, CivitaiMetaGrabber_url, re.MULTILINE)
            CivitaiMetaGrabber_url = f"https://civitai.com/api/v1/images?modelId={matches.group(4)}&modelVersionId={matches.group(5)}"

        if CivitaiMetaGrabber_model_id > 0:
            CivitaiMetaGrabber_url = f"https://civitai.com/api/v1/images?modelId={CivitaiMetaGrabber_model_id}"
        if '&modelVersionId=' not in str(CivitaiMetaGrabber_url):
            if CivitaiMetaGrabber_model_version_id > 0:
                CivitaiMetaGrabber_url += f"&modelVersionId={CivitaiMetaGrabber_model_version_id}"

        if '&sort=' not in str(CivitaiMetaGrabber_url):
            CivitaiMetaGrabber_url += '&sort=Most%20Comments'
        if '&limit=' not in str(CivitaiMetaGrabber_url):
            CivitaiMetaGrabber_url += f'&limit={CivitaiMetaGrabber_limit}'
        if '&page=' not in str(CivitaiMetaGrabber_url):
            CivitaiMetaGrabber_url += f'&page={CivitaiMetaGrabber_page}'

        log.warning(f"[###][Auto-LLM][CivitaiMetaGrabber] Grabber_URL: {CivitaiMetaGrabber_url}")
        # https: // civitai.com / api / v1 / images?limit = 10 & modelId = 85691
        completion = requests.get(CivitaiMetaGrabber_url, headers=headers).json()
        self.webpage_walker_array.clear()

        gallery_arr = []
        for index, ele in enumerate(completion['items']):
            x1 = ele['url'] or ""
            x2 = 'https://civitai.com/images/' + str(ele['id']) or ""
            ele2 = ele.get('meta') or {}
            x3 = ele2.get('prompt') or 'not include'
            x4 = ele2.get(CivitaiMetaGrabber_target_tag) or 'not include'
            x4 = striphtml(x4)
            result.append(x1)
            self.webpage_walker_array.append([x3, x4, x1, x2])
            gallery_arr.append((x1, x3))

            log.warning(f"[{index}][][auto_prompt_getter]URL: {x2} IMG: {x1}")
        model_idx = 0
        model_version_idx = 0
        try:
            model_idx = re.search(r"modelId=(\d+)", CivitaiMetaGrabber_url, re.IGNORECASE | re.MULTILINE).group(1)
            log.warning(f"[][][model_idx]{model_idx}")
            model_version_idx = re.search(r"modelVersionId=(\d+)", CivitaiMetaGrabber_url,
                                          re.IGNORECASE | re.MULTILINE).group(1)
            log.warning(f"[][][model_version_idx]{model_version_idx}")
        except AttributeError:
            log.warning(f"[][][AttributeError]model_idx={model_idx} model_version_idx={model_version_idx}", )

        return self.webpage_walker_array, gallery_arr, CivitaiMetaGrabber_url, int(model_idx), int(model_version_idx)

    def call_llm_mix(self, llm_apikey, json_str_x, llm_apiurl, llm_api_model_name):
        result_mix = ''
        url_append = ''
        if llm_apiurl.endswith('/'):
            llm_apiurl = llm_apiurl[:-1]
        if 'google' in llm_apiurl:
            llm_apiurl += f'/models/{llm_api_model_name}:generateContent'
            headers_x = {
                'Content-Type': 'application/json',
                'x-goog-api-key': f'{llm_apikey}'
            }
        else:
            llm_apiurl += '/chat/completions'
            headers_x = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {llm_apikey}',
            }
        log.warning(f"[][AutoLLM][getReq][Header]{headers_x}")

        try:
            #lm-studio   http://localhost:1234/v1/chat/completions
            #ollama      http://localhost:11434/v1/chat/completions
            #google      https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent
            log.warning(f"[Auto-LLM][][]Req URL=> {llm_apiurl}")
            completion = requests.post(llm_apiurl, headers=headers_x, json=json_str_x)
            completion_json = completion.json()
            if 'google' in llm_apiurl:
                result_mix = completion_json['candidates'][0]['content']['parts'][0]['text']
            else:
                result_mix = completion_json['choices'][0]['message']['content']
            log.warning("[Auto-LLM][][]Server Ans=> " + result_mix)

        except Exception as e:
            e = str(e)
            self.llm_history_array.append([e, e, e, e])
            result_mix = "[Auto-LLM][Result][Missing LLM-Text]" + e
            log.warning("[Auto-LLM][][]Missing LLM Server?" + e)
        result_mix = result_mix.replace('\n', ' ')
        return result_mix

    def call_llm_eye_open(self, llm_is_enabled, llm_recursive_use, llm_keep_your_prompt_use,
                          llm_text_system_prompt, llm_text_ur_prompt,
                          llm_text_max_token, llm_text_tempture,
                          llm_apiurl, llm_apikey, llm_api_model_name,
                          llm_api_translate_system_prompt, llm_api_translate_enabled,
                          llm_is_open_eye,
                          llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_ur_prompt_image_eye,
                          llm_text_tempture_eye,
                          llm_text_max_token_eye,
                          llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                          llm_post_action_cmd_feedback_type,
                          llm_post_action_cmd,
                          llm_top_k_text, llm_top_p_text, llm_top_k_vision, llm_top_p_vision,
                          llm_loop_enabled, llm_loop_ur_prompt, llm_loop_count_slider, llm_loop_each_append,
                          CivitaiMetaGrabber_to_llm_text_ur_prompt, CivitaiMetaGrabber_to_prompt):
        llm_before_action_cmd_return_value = self.do_subprocess_action(llm_before_action_cmd)

        if llm_recursive_use and (self.llm_history_array.__len__() > 1):
            llm_text_ur_prompt = (llm_text_ur_prompt if llm_keep_your_prompt_use else "") + " " + \
                                 self.llm_history_array[self.llm_history_array.__len__() - 1][1]
        base64_image = ""

        try:
            # if type(llm_text_ur_prompt_image_eye) == str:
            #     process_buffer = open(llm_text_ur_prompt_image_eye, "rb").read()
            # else:
            # log.warning(f"[][][call_llm_eye_open]PIL Image llm_text_ur_prompt_image_eye.format: {llm_text_ur_prompt_image_eye} ")
            # xprint(llm_text_ur_prompt_image_eye)
            process_buffer = BytesIO()
            llm_text_ur_prompt_image_eye.save(process_buffer, format='PNG')
            base64_image = base64.b64encode(process_buffer.getvalue()).decode("utf-8")
            # print("[][call_llm_eye_open][]base64_image", base64_image)

        except Exception as e:
            log.error(f"[][][call_llm_eye_open]PIL Image Error: {e} ")
            self.llm_history_array.append(["missing input image ?", e, e, e])
            # return "[][call_llm_eye_open]missing input image ?" + e, self.llm_history_array
            return "missing input image ?", self.llm_history_array
        try:
            # self.check_api_uri(llm_apiurl, llm_apikey)
            json_x0 = getReqJson(llm_apiurl, llm_apikey, llm_api_model_name, llm_text_system_prompt,
                                 llm_text_ur_prompt,
                                 llm_text_tempture, llm_top_k_text, llm_top_p_text, llm_text_max_token,
                                 base64_image,
                                 llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_max_token_eye,
                                 llm_text_tempture_eye, llm_top_p_vision)
            result_text = self.call_llm_mix(llm_apikey, json_x0, llm_apiurl, llm_api_model_name)
            llm_answers_array = []
            if llm_loop_enabled:
                llm_loop_ur_prompt_array = llm_loop_ur_prompt.split('\n')

                for i in range(llm_loop_count_slider):
                    json_x2 = getReqJson(llm_apiurl, llm_apikey, llm_api_model_name, llm_text_system_prompt,
                                         llm_loop_ur_prompt_array[
                                             min(len(llm_loop_ur_prompt_array) - 1, i)] + result_text,
                                         llm_text_tempture, llm_top_k_text, llm_top_p_text, llm_text_max_token, '',
                                         llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_max_token_eye,
                                         llm_text_tempture_eye,
                                         llm_top_p_vision)

                    llm_answers_array.append(self.call_llm_mix(llm_apikey, json_x2, llm_apiurl, llm_api_model_name))
        except Exception as e:
            e = str(e)
            log.error(f"[][][call_llm_eye_open]Model Error: {e}")
            self.llm_history_array.append([e, e, e, e])
            return e, self.llm_history_array

        result = result_text.replace('\n', ' ')
        result_translate = ""
        if llm_api_translate_enabled:
            result_translate = self.call_llm_translate(llm_apiurl, llm_apikey, llm_api_model_name,
                                                       llm_api_translate_system_prompt, result,
                                                       llm_text_max_token_eye)
        self.llm_history_array.append([result, llm_text_system_prompt_eye, llm_text_ur_prompt_eye, result_translate])
        if len(self.llm_history_array) > 3:
            self.llm_history_array.remove(self.llm_history_array[0])
        print("[][auto-llm][call_llm_eye_open] ", result)

        self.do_subprocess_action(llm_post_action_cmd)

        return result, self.llm_history_array

    def justRtn(self, r):
        return r

    def call_llm_text(self, llm_is_enabled, llm_recursive_use, llm_keep_your_prompt_use,
                      llm_text_system_prompt, llm_text_ur_prompt,
                      llm_text_max_token, llm_text_tempture,
                      llm_apiurl, llm_apikey, llm_api_model_name,
                      llm_api_translate_system_prompt, llm_api_translate_enabled,
                      llm_is_open_eye,
                      llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_ur_prompt_image_eye,
                      llm_text_tempture_eye,
                      llm_text_max_token_eye,
                      llm_before_action_cmd_feedback_type, llm_before_action_cmd,
                      llm_post_action_cmd_feedback_type,
                      llm_post_action_cmd,
                      llm_top_k_text, llm_top_p_text, llm_top_k_vision, llm_top_p_vision,
                      llm_loop_enabled, llm_loop_ur_prompt, llm_loop_count_slider, llm_loop_each_append,
                      CivitaiMetaGrabber_to_llm_text_ur_prompt, CivitaiMetaGrabber_to_prompt
                      ):

        llm_before_action_cmd_return_value = self.do_subprocess_action(llm_before_action_cmd)
        if EnumCmdReturnType.LLM_USER_PROMPT.value in llm_before_action_cmd_feedback_type:
            llm_text_ur_prompt += llm_before_action_cmd_return_value

        if llm_recursive_use and (self.llm_history_array.__len__() > 1):
            llm_text_ur_prompt = (llm_text_ur_prompt if llm_keep_your_prompt_use else "") + " " + \
                                 self.llm_history_array[self.llm_history_array.__len__() - 1][1]
        # self.check_api_uri(llm_apiurl, llm_apikey)
        result_text = ''

        if CivitaiMetaGrabber_to_llm_text_ur_prompt:
            llm_text_ur_prompt += self.getRandomPrompt()
            # llm_text_ur_prompt += self.webpage_walker_array[0][2]
        try:

            json_x1 = getReqJson(llm_apiurl, llm_apikey, llm_api_model_name, llm_text_system_prompt, llm_text_ur_prompt,
                                 llm_text_tempture, llm_top_k_text, llm_top_p_text, llm_text_max_token, '',
                                 llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_max_token_eye,
                                 llm_text_tempture_eye,
                                 llm_top_p_vision)

            result_text = self.call_llm_mix(llm_apikey, json_x1, llm_apiurl, llm_api_model_name)
            llm_answers_array = []
            if llm_loop_enabled:
                llm_loop_ur_prompt_array = llm_loop_ur_prompt.split('\n')

                for i in range(llm_loop_count_slider):
                    json_x2 = getReqJson(llm_apiurl, llm_apikey, llm_api_model_name, llm_text_system_prompt,
                                         llm_loop_ur_prompt_array[
                                             min(len(llm_loop_ur_prompt_array) - 1, i)] + result_text,
                                         llm_text_tempture, llm_top_k_text, llm_top_p_text, llm_text_max_token, '',
                                         llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_max_token_eye,
                                         llm_text_tempture_eye,
                                         llm_top_p_vision)

                    llm_answers_array.append(self.call_llm_mix(llm_apikey, json_x2, llm_apiurl, llm_api_model_name))

        except Exception as e:
            e = str(e)
            self.llm_history_array.append([e, e, e, e])
            return e, self.llm_history_array
        # result = completion.choices[0].message.content
        if llm_loop_each_append:
            result = " | ".join(llm_answers_array)

        result = result_text.replace('\n', ' ')
        result_translate = ""
        if llm_api_translate_enabled:
            result_translate = self.call_llm_translate(llm_apiurl, llm_apikey, llm_api_model_name,
                                                       llm_api_translate_system_prompt, result,
                                                       llm_text_max_token)

        self.llm_history_array.append([result, llm_text_ur_prompt, llm_text_system_prompt, result_translate])
        if len(self.llm_history_array) > 3:
            self.llm_history_array.remove(self.llm_history_array[0])

        # print("[][auto-llm][call_llm_pythonlib] ", result, result_translate)
        self.do_subprocess_action(llm_post_action_cmd)
        return result, self.llm_history_array

    def do_subprocess_action(self, llm_post_action_cmd):
        if llm_post_action_cmd.__len__() <= 0:
            return ""
        p = subprocess.Popen(llm_post_action_cmd.split(" "), text=True, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (out, err) = p.communicate()
        ret = p.wait()
        ret = True if ret == 0 else False
        if ret:
            log.warning("Command succeeded. " + llm_post_action_cmd + " output=" + out)
            self.llm_history_array.append(["[O]PostAction-Command succeeded.", err, llm_post_action_cmd, out])
        else:
            log.warning("Command failed. " + llm_post_action_cmd + " err=" + err)
            self.llm_history_array.append(["[X]PostAction-Command failed.", err, llm_post_action_cmd, out])
        return out

    def call_llm_translate(self, llm_apiurl, llm_apikey, llm_api_model_name, llm_api_translate_system_prompt,
                           llm_api_translate_user_prompt,
                           _llm_text_max_token):
        try:

            json_x3 = {
                'model': f"{llm_api_model_name}",
                'contents': [
                    {"role": "user", "parts": [{"text": f"{llm_api_translate_user_prompt}"}]}
                ],
                'messages': [
                    {"role": "system", "content": llm_api_translate_system_prompt},
                    {"role": "user", "content": llm_api_translate_user_prompt}
                ],
                'max_tokens': _llm_text_max_token,
                "max_output_tokens": _llm_text_max_token,
                'temperature': 0.2,
            }
        except Exception as e:
            e = str(e)
            log.error(f"[][][call_llm_pythonlib]Error: {e}")
            return e
        result_translate = self.call_llm_mix(llm_apikey, json_x3, llm_apiurl, llm_api_model_name)

        result_translate = result_translate.replace('\n', '').encode("utf-8").decode()
        log.warning(f"[][][call_llm_translate]: {result_translate}")
        return result_translate

    def ui(self, is_img2img):
        # print("\n\n[][Init-UI][sd-webui-prompt-auto-llm]: " + str(is_img2img) + "\n\n")
        # log.error(f'[][][][][][Init-UI][sd-webui-prompt-auto-llm]: " + str({is_img2img})')
        examples = [
            ["The Moon's orbit around Earth has"],
            ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
        ]
        default_llm_ans_json = {"llm-text-ans": "", "llm-vision-ans": ""}

        with gr.Blocks():
            # gr.Markdown("Blocks")
            with gr.Accordion(open=False, label="Auto LLM v20240829"):
                with gr.Tab("LLM-text"):
                    # with gr.Accordion(open=True, label="[Prompt]/[LLM-PythonLib]"):
                    gr.Markdown("* Generate forever mode \n"
                                "* Story board mode")
                    llm_is_enabled = gr.Checkbox(label=" Enable LLM-Answer to SD-prompt", value=False)

                    llm_keep_your_prompt_use = gr.Checkbox(label=" Keep LLM-Your-Prompt ahead each request",
                                                           value=False)

                    with gr.Row():
                        with gr.Column(scale=1):
                            llm_text_system_prompt = gr.Textbox(label="1. [LLM-System-Prompt]", lines=5,
                                                                value=self.llm_sys_text_template,
                                                                placeholder=self.llm_sys_text_template
                                                                )
                            llm_text_ur_prompt = gr.Textbox(label="2. [LLM-Your-Prompt]", lines=8,
                                                            value="A superstar on stage.",
                                                            placeholder="A superstar on stage.")
                        with gr.Column(scale=4):
                            llm_text_tempture = gr.Slider(-2, 2, value=0.5, step=0.01,
                                                          label="3.1 LLM temperature", elem_id="llm_text_tempture",
                                                          interactive=True,
                                                          #hint='temperature (Deterministic) | (More creative)'
                                                          )
                            with gr.Row():
                                llm_top_k_text = gr.Slider(
                                    elem_id="llm_top_k_text", label="3.2 LLM Top k ", value=8, minimum=1, maximum=20,
                                    step=0.01,
                                    interactive=True,
                                    #hint='Strategy is to sample from a shortlist of the top K tokens. This approach allows the other high-scoring tokens a chance of being picked.'
                                )
                                llm_top_p_text = gr.Slider(
                                    elem_id="llm_top_p_text", label="3.3 LLM Top p ", value=0.9, minimum=0, maximum=1,
                                    step=0.01,
                                    interactive=True,
                                    #hint=' (nucleus): The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus.'
                                )

                            llm_text_max_token = gr.Slider(5, 5000, value=150, step=5,
                                                           label="3.4 LLM Max length(tokens)")
                            self.llm_llm_answer = gr.Textbox(
                                # inputs=[llm_ans_state['llm-text-ans']],
                                show_copy_button=True, interactive=True,
                                label="4. [LLM-Answer]", lines=6, placeholder="LLM says.")

                            with gr.Row():
                                llm_sendto_txt2img = gr.Button("send to txt2img")
                                llm_sendto_img2img = gr.Button("send to img2img")

                    llm_button = gr.Button("Call LLM above")
                    llm_history = gr.Dataframe(
                        interactive=True,
                        wrap=True,
                        label="History/StoryBoard",
                        headers=["llm_answer", "system_prompt", "ur_prompt", "result_translate"],
                        datatype=["str", "str", "str", "str"],
                        row_count=3,
                        col_count=(4, "fixed"),
                    )

                with gr.Tab("LLM-vision"):
                    llm_is_open_eye = gr.Checkbox(label="Enable LLM-vision👀", value=False)
                    llm_is_open_eye_last_one_image = gr.Checkbox(
                        label="Auto look at last one image (auto put last image into Step.2)",
                        value=True)
                    with gr.Row():
                        with gr.Column(scale=1):
                            llm_text_system_prompt_eye = gr.Textbox(label=" 1.[LLM-System-Prompt-eye]", lines=13,
                                                                    value=self.llm_sys_vision_template,
                                                                    placeholder=self.llm_sys_vision_template)
                            llm_text_ur_prompt_eye = gr.Textbox(label=" 2.[Your-prompt-eye]", lines=13,
                                                                value="What’s in this image?",
                                                                placeholder="What’s in this image?")
                        with gr.Column(scale=4):
                            llm_text_ur_prompt_image_eye = gr.Image(label="2. [Your-Image]", lines=1, type='pil')
                            llm_text_tempture_eye = gr.Slider(-2, 2, value=0.1, step=0.01,
                                                              label="3.1 LLM temperature (Deterministic) | (More creative)")
                            with gr.Row():
                                llm_top_k_vision = gr.Slider(
                                    elem_id="llm_top_k_vision", label="3.2 LLM Top k ", value=8, minimum=1, maximum=20,
                                    step=0.01,
                                    interactive=True,
                                    #hint='Strategy is to sample from a shortlist of the top K tokens. This approach allows the other high-scoring tokens a chance of being picked.'
                                )
                                llm_top_p_vision = gr.Slider(
                                    elem_id="llm_top_p_vision", label="3.3 LLM Top p ", value=0.9, minimum=0, maximum=1,
                                    step=0.01,
                                    interactive=True,
                                    #hint=' (nucleus): The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus.'
                                )
                            llm_text_max_token_eye = gr.Slider(5, 5000, value=150, step=5,
                                                               label="3.4 LLM Max length(tokens)")
                            llm_llm_answer_eye = gr.Textbox(inputs=self.process, show_copy_button=True,
                                                            interactive=True,
                                                            label="4. [LLM-Answer-eye]", lines=6,
                                                            placeholder="LLM says.")
                            with gr.Row():
                                llm_sendto_txt2img_vision = gr.Button("send to txt2img")
                                llm_sendto_img2img_vision = gr.Button("send to img2img")
                    llm_button_eye = gr.Button("Call LLM-vision above")
                    llm_history_eye = gr.Dataframe(
                        interactive=True,
                        wrap=True,
                        label="History/StoryBoard",
                        headers=["llm_answer", "system_prompt", "ur_prompt", "result_translate"],
                        datatype=["str", "str", "str", "str"],
                        row_count=3,
                        col_count=(4, "fixed"),
                    )

                # with gr.Tab("LLM-through-embeddings"):
                #     llm_is_through = gr.Checkbox(label="Enable LLM-through", value=False)
                #
                # with gr.Tab("LLM-asking-chat"):
                #     llm_is_asking = gr.Checkbox(label="Enable asking", value=False)
                # with gr.Tab("Gallery"):
                #     gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=3, rows=1, object_fit="contain", height="auto")
                with gr.Tab("LLM-ask-LLM"):
                    gr.Markdown("* LLM -> LLM-line-1 -> LLM-line-2 -> LLM-line-3 -> SD\n"
                                "* digging into deep of model\n"
                                "* model suggest 7B\n"
                                "* 2.Append each. ==> u will get every line llm-answer separately.  \n"
                                "* function discussions:  https://github.com/xlinx/sd-webui-decadetw-auto-prompt-llm/discussions/14 \n"
                                )
                    llm_recursive_use = gr.Checkbox(label="0. LLM-ask-LLM/Recursive-prompt. (Prompt from last LLM-Ans)🌀",
                                                    value=False)
                    llm_loop_enabled = gr.Checkbox(label="1. Append follow textarea each-line as feature to prompt", value=False)
                    llm_loop_each_append = gr.Checkbox(
                        label="2.Append follow each line LLM-Ans. [ uncheck:Send last one LLM-Answer. ] [ check:Append each LLM-Answer ]",
                        value=False)
                    llm_loop_count_slider = gr.Slider(1, 5, value=1, step=1,
                                                      label="2. LLM-Loop Count (1=> append 1 more times LLM-Text. calling LLM total is 2)")
                    llm_loop_ur_prompt = gr.Textbox(
                        label="3. option.[LLM-Loop-Your-Prompt] line by line append to every loop",
                        lines=3,
                        value="red\nyellow\nblue",
                        placeholder="red\nyellow\nblue")
                with gr.Tab("Setup"):
                    gr.Markdown("* API-URI: LMStudio=>http://localhost:1234/v1 \n"
                                "* API-URI: ollama  => http://localhost:11434/v1 \n"
                                "* API-ModelName: LMStudio can be empty here is fine; select it LMStudio App; ollama should set like: llama3.1 (cmd:ollama list)\n"
                                "* OLLAMA OpenAI compatibility https://ollama.com/blog/openai-compatibility\n"
                                )
                    llm_apiurl = gr.Textbox(
                        label="1.[LLM-URL]  ",
                        lines=1,
                        value="http://localhost:1234/v1")
                    llm_apiurl_radio = gr.Radio(Enum_Api_URL_ReturnType.values(),
                                                value=Enum_Api_URL_ReturnType.values()[0],
                                                label="1.1 Quick URL"
                                                )
                    llm_apikey = gr.Textbox(label="2.[LLM-API-Key] lm-studio | ollama", lines=1, value="lm-studio")
                    llm_api_model_name = gr.Textbox(
                        label="3.[LLM-Model-Name] its no need to set for lmstudio; others like:ollama gemini need. ",
                        lines=1,
                        value="llama3.1",
                        placeholder="llama3.1, llama2, gemma2 gemini-pro gemini-1.5-flash(vision)")
                    llm_api_model_name_radio = gr.Radio(Enum_Api_MODEL_ReturnType.values(),
                                                        value=Enum_Api_URL_ReturnType.values()[0],
                                                        label="3.1 Quick MODEL"
                                                        )
                    with gr.Row():
                        with gr.Column(scale=2):
                            llm_before_action_cmd_feedback_type = gr.Radio(EnumCmdReturnType.values(),
                                                                           value='just-call',
                                                                           label="4.1 Return value type",
                                                                           info="Capture CMD return value for LLM-xxx?")
                            # llm_before_action_cmd_feedback = gr.Checkbox(label="Capture CMD return value to LLM-user-prompt",value=False)
                            # llm_before_action_cmd_feedback_vision = gr.Checkbox(label="Capture CMD return image path to LLM-vision",value=False)
                            llm_before_action_cmd = gr.Textbox(
                                label="4.1 Before LLM API call action",
                                lines=1,
                                value="",
                                placeholder="""run getStoryLine.bat | sh myStoryBook.sh'""",
                                info="call ur script(.bat, .sh) ")

                        with gr.Column(scale=2):
                            llm_post_action_cmd_feedback_type = gr.Radio(EnumCmdReturnType.values(), value='just-call',
                                                                         label="4.2 Return value type",
                                                                         info="Capture CMD return value for LLM-xxx?")
                            # llm_post_action_cmd_feedback = gr.Checkbox(
                            #     label="Capture CMD return value to LLM-user-prompt",
                            #     info="You can call ur script(.bat, .sh) for LLM prompt or call customer cmd. ex: curl",
                            #     value=False)
                            # llm_post_action_cmd_feedback_vision = gr.Checkbox(
                            #     label="Capture CMD return image path to LLM-vision",
                            #     info="call ur script(.bat, .sh) for LLM-vision image input path",
                            #     value=False, enable=False)
                            llm_post_action_cmd = gr.Textbox(
                                label="4.2 After LLM API call action ",
                                lines=1,
                                value="",
                                placeholder="""curl http://localhost:11434/api/generate -d '{"keep_alive": 0}'""",
                                info="call ur script(.bat, .sh) ")

                    llm_api_translate_enabled = gr.Checkbox(
                        label="Enable translate LLM-answer to Your language.(won`t effect with SD, just for reference. )",
                        value=False)
                    llm_api_translate_system_prompt = gr.Textbox(label=" 5.[LLM-Translate-System-Prompt]", lines=5,
                                                                 value=self.llm_sys_translate_template)
                with gr.Tab("Civitai Meta Grabber"):
                    gr.Markdown("- Quick walk through all image style prompt of model that u never download\n"
                                "   - second-model(no need download just ref prompt) for generate with ur(main-model) 1girl. \n"
                                "   - then u should get 1girl with indoor-design and describe background in-detail quickly\n"
                                "   - auto Find all images meta(prompt...) under this model \n"
                                "- API manual = https://civitai.com/api/v1/images\n"
                                "   - API URL example: https://civitai.com/api/v1/images?modelId=85691\n"
                                "   - if model have multi version, ex: https://civitai.com/api/v1/images?limit=100&modelId=43331&modelVersionId=176425 \n"
                                "- [NEW] Civitai AIR ‐ Uniform Resource Names for AI\n"
                                "   - manual: https://github.com/civitai/civitai/wiki/AIR-%E2%80%90-Uniform-Resource-Names-for-AI\n"
                                "   - ***How find AIR. look every model detail bottom of side menu.(click copy)\n"
                                "   - ***Civitai AIR look like this=> urn:air:sd1:checkpoint:civitai:85691@93152\n"
                                )
                    CivitaiMetaGrabber_to_llm_text_ur_prompt = gr.Checkbox(
                        label="Random Pick (2.customer_var) to LLM-text-ur-prompt (need enable LLM-text)",
                        value=False)
                    CivitaiMetaGrabber_to_prompt = gr.Checkbox(
                        label="Random Pick (2.customer_var) to Prompt (No need enable LLM-text just append prompt without ask LLM)",
                        value=False)
                    auto_prompt_getter_remove_lora_tag = gr.Checkbox(
                        label="Remove Lora tag in prompt (any char inside <***>)", value=True)

                    CivitaiMetaGrabber_url = gr.Textbox(
                        label="1 URL or Civitai-AIR ( https://civitai.com/api/v1/images?modelId=85691 ) or ( urn:air:sd1:checkpoint:civitai:85691@93152 )",
                        lines=1,
                        value="https://civitai.com/api/v1/images?modelId=85691",
                        placeholder="https://civitai.com/api/v1/images?modelId=85691",
                        info="")
                    with gr.Row():
                        CivitaiMetaGrabber_model_id = gr.Slider(0, 999999, value=0, step=1,
                                                                label="1.1 Civitai Model ID(0 use above URL setting)",
                                                                interactive=True)
                        CivitaiMetaGrabber_model_version_id = gr.Slider(0, 999999, value=0, step=1,
                                                                        label="1.2 Civitai Model Version ID(0 use above URL setting)",
                                                                        interactive=True)
                    CivitaiMetaGrabber_target_tag = gr.Textbox(
                        label="2.1 customer_var (prompt | negativePrompt | comfy | ...)(pick var left side menu https://civitai.com/search/images?query=realistic)",
                        lines=1,
                        value="prompt",
                        placeholder="prompt negativePrompt id url hash width nsfw nsfwLevel createAt...",
                        info="")
                    with gr.Row():
                        CivitaiMetaGrabber_limit = gr.Slider(1, 100, value=10, step=1,
                                                             label="2.3 How many result per page", interactive=True)
                        CivitaiMetaGrabber_page = gr.Slider(1, 100, value=1, step=1,
                                                            label="2.2 Model Image Gallery Page NO.", interactive=True)
                    # CivitaiMetaGrabber_model_id = gr.Slider(1, 100, value=1, step=1,
                    #                                     label="2.2 Model Image Gallery Page", interactive=True)
                    # CivitaiMetaGrabber_model_version_id = gr.Slider(1, 100, value=1, step=1,
                    #                                     label="2.2 Model Image Gallery Page", interactive=True)
                    CivitaiMetaGrabber_go_button = gr.Button("Click grabber list first ")
                    CivitaiMetaGrabber_gallery = gr.Gallery(label="Generated images", show_label=False,
                                                            elem_id="gallery", columns=3, rows=1, object_fit="contain")
                    CivitaiMetaGrabber_history = gr.Dataframe(
                        interactive=True,
                        wrap=True,
                        label="List all Meta in this model",
                        headers=["prompt", "customer_var", "image_url", "post_url"],
                        datatype=["str", "str", "str", "str"],
                        row_count=3,
                        col_count=(4, "fixed"),
                    )
                with gr.Tab("Export/Import"):
                    gr.Markdown("* Share and see how people how to use LLM in SD.\n"
                                "* Community Share Link: \n"
                                "* https://github.com/xlinx/sd-webui-decadetw-auto-prompt-llm/discussions/12\n"
                                )
                    with gr.Row():
                        community_export_btn = gr.Button("0. Export/Save to Disk|Text")
                        community_import_btn = gr.Button("0. Import from Disk|Text")

                    community_text = gr.Textbox(
                        label="1. copy/paste Text-LLM-Setting here",
                        lines=3,
                        value="",
                        placeholder="Export&Save first; if here empty will load from disk")

        all_var_val = [llm_is_enabled, llm_recursive_use, llm_keep_your_prompt_use,
                       llm_text_system_prompt, llm_text_ur_prompt,
                       llm_text_max_token, llm_text_tempture,
                       llm_apiurl, llm_apikey, llm_api_model_name,
                       llm_api_translate_system_prompt, llm_api_translate_enabled,
                       llm_is_open_eye,
                       llm_text_system_prompt_eye, llm_text_ur_prompt_eye, llm_text_ur_prompt_image_eye,
                       llm_text_tempture_eye,
                       llm_text_max_token_eye,
                       llm_before_action_cmd_feedback_type, llm_before_action_cmd, llm_post_action_cmd_feedback_type,
                       llm_post_action_cmd,
                       llm_top_k_text, llm_top_p_text, llm_top_k_vision, llm_top_p_vision,
                       llm_loop_enabled, llm_loop_ur_prompt, llm_loop_count_slider, llm_loop_each_append,
                       CivitaiMetaGrabber_to_llm_text_ur_prompt, CivitaiMetaGrabber_to_prompt
                       ]
        llm_apiurl_radio.change(self.justRtn, inputs=llm_apiurl_radio, outputs=llm_apiurl)
        llm_api_model_name_radio.change(self.justRtn, inputs=llm_api_model_name_radio, outputs=llm_api_model_name)

        CivitaiMetaGrabber_go_button.click(self.CMG_getter,
                                           inputs=[CivitaiMetaGrabber_url, CivitaiMetaGrabber_target_tag,
                                                   CivitaiMetaGrabber_page, CivitaiMetaGrabber_limit,
                                                   CivitaiMetaGrabber_model_id, CivitaiMetaGrabber_model_version_id],
                                           outputs=[CivitaiMetaGrabber_history, CivitaiMetaGrabber_gallery,
                                                    CivitaiMetaGrabber_url, CivitaiMetaGrabber_model_id,
                                                    CivitaiMetaGrabber_model_version_id])
        community_export_btn.click(community_export_to_text,
                                   inputs=all_var_val,
                                   outputs=[community_text])
        community_import_btn.click(community_import_from_text,
                                   inputs=community_text,
                                   outputs=all_var_val)
        llm_button_eye.click(self.call_llm_eye_open,
                             inputs=all_var_val,
                             outputs=[llm_llm_answer_eye, llm_history_eye])
        llm_button.click(self.call_llm_text,
                         inputs=all_var_val,
                         outputs=[self.llm_llm_answer, llm_history])

        llm_sendto_txt2img.click(fn=None, _js="function(prompt){sendPromptAutoPromptLLM('txt2img', prompt)}",
                                 inputs=[self.llm_llm_answer])
        llm_sendto_img2img.click(fn=None, _js="function(prompt){sendPromptAutoPromptLLM('img2img', prompt)}",
                                 inputs=[self.llm_llm_answer])

        for e in [self.llm_llm_answer, llm_history, llm_llm_answer_eye, llm_history_eye]:
            e.do_not_save_to_config = True

        return all_var_val

    # def process(self, p: StableDiffusionProcessingTxt2Img,*args):
    def postprocess(self, p: StableDiffusionProcessingTxt2Img, *args):
        log.warning(f"_____[AUTO_LLM][postprocess][] ")
        # self.llm_ans_state.value = 'v1'
        # self.llm_ans_state.update({'value': 'v11'})
        # self.llm_ans_state.update(value='v111')
        # gr.update(self.llm_ans_state, value='v1111')
        #
        # self.llm_llm_answer.value = 'v2'
        # self.llm_llm_answer.update(value='v22')
        # gr.update(self.llm_llm_answer, value='v222')

    def getRandomPrompt(self):
        lenx = len(self.webpage_walker_array)
        if lenx < 1:
            log.warning(f"_____[AUTO_LLM][getRandomPrompt][]Len<1. u need click grabber button first")
            return ''
        g_result = self.webpage_walker_array[random.randrange(0, lenx, 1)][1]
        count = 0
        while count < 10 and len(g_result) < 20:
            g_result = self.webpage_walker_array[random.randrange(0, lenx, 1)][1]
            count += 1
        log.warning(f"_____[AUTO_LLM][getRandomPrompt][]CivitaiMetaGrabber={g_result}")

        return g_result

    def process(self, p: StableDiffusionProcessingTxt2Img, *args):
        global args_dict
        args_dict = dict(zip(all_var_key, args))
        # if llm_is_enabled:
        if args_dict.get('llm_is_enabled'):
            r = self.call_llm_text(*args)
            g_result = str(r[0])

            # g_result += g_result+"\n\n"+translate_r
            for i in range(len(p.all_prompts)):
                p.all_prompts[i] = p.all_prompts[i] + (",\n" if (p.all_prompts[0].__len__() > 0) else "\n") + g_result

        if args_dict.get('CivitaiMetaGrabber_to_prompt'):
            g_result = self.getRandomPrompt()
            for i in range(len(p.all_prompts)):
                p.all_prompts[i] = p.all_prompts[i] + (",\n" if (p.all_prompts[0].__len__() > 0) else "\n") + g_result
        # if llm_is_open_eye:
        if args_dict.get('llm_is_open_eye'):
            r2 = self.call_llm_eye_open(*args)
            g_result2 = str(r2[0])
            for i in range(len(p.all_prompts)):
                p.all_prompts[i] = p.all_prompts[i] + (",\n" if (p.all_prompts[0].__len__() > 0) else "\n") + g_result2

        return p.all_prompts[0]


args_dict = None
all_var_key = ['llm_is_enabled', 'llm_recursive_use', 'llm_keep_your_prompt_use',
               'llm_text_system_prompt', 'llm_text_ur_prompt',
               'llm_text_max_token', 'llm_text_tempture',
               'llm_apiurl', 'llm_apikey', 'llm_api_model_name',
               'llm_api_translate_system_prompt', 'llm_api_translate_enabled',
               'llm_is_open_eye',
               'llm_text_system_prompt_eye', 'llm_text_ur_prompt_eye', 'llm_text_ur_prompt_image_eye',
               'llm_text_tempture_eye',
               'llm_text_max_token_eye',
               'llm_before_action_cmd_feedback_type', 'llm_before_action_cmd', 'llm_post_action_cmd_feedback_type',
               'llm_post_action_cmd',
               'llm_top_k_text', 'llm_top_p_text', 'llm_top_k_vision', 'llm_top_p_vision',
               'llm_loop_enabled', 'llm_loop_ur_prompt', 'llm_loop_count_slider', 'llm_loop_each_append',
               'CivitaiMetaGrabber_to_llm_text_ur_prompt', 'CivitaiMetaGrabber_to_prompt'
               ]
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
# path_maps = {
#     "txt2img": opts.outdir_samples or opts.outdir_txt2img_samples,
#     "img2img": opts.outdir_samples or opts.outdir_img2img_samples,
#     "txt2img-grids": opts.outdir_grids or opts.outdir_txt2img_grids,
#     "img2img-grids": opts.outdir_grids or opts.outdir_img2img_grids,
#     "Extras": opts.outdir_samples or opts.outdir_extras_samples
# }
# https: // platform.openai.com / docs / guides / vision?lang = curl
# if EnumCmdReturnType.LLM_VISION_IMG_PATH.value in llm_before_action_cmd_feedback_type:
#     llm_text_ur_prompt_image_eye = llm_before_action_cmd_return_value
# script_callbacks.on_ui_tabs(on_ui_tabs )
#https://platform.openai.com/docs/api-reference/introduction
