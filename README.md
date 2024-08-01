# sd-webui-decadetw-auto-prompt-llm
* Automatic1111 extension 
* Calling LLM; auto prompt for batch generate images 
* embedding LLM into prompts.
* You will notice the image will have content.

## Motivation

* batch image generate with LLM
* prompt detail is important
* detail using LLM
* calling LLM by 
  * python lib (just fill LLM-prompt)
    * when generate forever mode
      * example as follows figure Red-box. 
      * just tell LLM who, when or what
      * LLM will take care details.
    * when a story-board mode (You can generate serial image follow a story by LLM context.)
      * example: 2~10 images(like comic book)
      * a superstar on stage
      * she is singing
      * people give her flower
      * ...etc.
  * Enable LLM vision (open LLM eye to see then SD-prompt)👀
    * https://huggingface.co/xtuner/llava-phi-3-mini-gguf
      * llava-phi-3-mini-mmproj-f16.gguf (600MB)
      * llava-phi-3-mini-f16.gguf (7G)
  * javascript fetch POST method (install Yourself )
    * security issue, but u can consider as follows 
    * https://github.com/pmcculler/sd-dynamic-javascript
    * https://github.com/ThereforeGames/unprompted
    * https://github.com/adieyal/sd-dynamic-prompts
    * https://en.wikipedia.org/wiki/Server-side_request_forgery
    * and Command Line Arg --allow-code


---


<table style="border-width:0px" >
 <tr>
    <td><b style="font-size:30px">1. sd-webui-prompt</b></td>
    <td><b style="font-size:30px">2. LLM-Your-prompt</b></td>
 </tr>
 <tr>
    <td>1girl</td>
    <td>a superstar on stage.</td>
 </tr>
<tr>
    <td colspan="2"><b style="font-size:30px">3. LLM will answer other detail</b></td>
 </tr>
<tr >
    <td colspan="2">The superstar, with their hair flowing in the wind, stands on the stage. The lights dance around them, creating a magical moment that fills everyone present with awe. Their eyes shine bright, as if they are ready to take on the world.</td>
 </tr>
<tr >
    <td colspan="2">The superstar stands tall in their sparkling costume, surrounded by fans who chant and cheer their name. The lights shine down on them, making their hair shine like silver. The crowd is electric, every muscle tense, waiting for the superstar to perform</td>
 </tr>
<tr >
    <td colspan="2">etc,.</td>
 </tr>
</table>
<table style="border-width:0px" >
 <tr>
    <td><b style="font-size:30px">LLM as text-assist&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b></td>
    <td><b style="font-size:30px">LLM as vision-assistant</b></td>
 </tr>
 <tr>
    <td><img src="images/readme0.png"></img></td>
    <td><img src="images/readme3.png"></img></td>
 </tr>
 <tr>
    <td colspan="2"><img src="images/readme1.png"></img></td>
 </tr>
</table>

## Installtion

### Suggestion LLM Model

* LLM (normal, chat, assistant)
  * 4B VRAM<2G
    * CHE-72/Qwen1.5-4B-Chat-Q2_K-GGUF/qwen1.5-4b-chat-q2_k.gguf
      * https://huggingface.co/CHE-72/Qwen1.5-4B-Chat-Q2_K-GGUF
  * 7B VRAM<8G
    * ccpl17/Llama-3-Taiwan-8B-Instruct-GGUF/Llama-3-Taiwan-8B-Instruct.Q2_K.gguf
    * Lewdiculous/L3-8B-Stheno-v3.2-GGUF-IQ-Imatrix/L3-8B-Stheno-v3.2-IQ3_XXS-imat.gguf
* Enable LLM vision 👀
    * https://huggingface.co/xtuner/llava-phi-3-mini-gguf
      * llava-phi-3-mini-mmproj-f16.gguf (600MB)
      * llava-phi-3-mini-f16.gguf (7G)
### Suggestion software

* https://lmstudio.ai/ (windows)
* https://ollama.com/ (mac, linux)
* https://github.com/openai/openai-python

<img src="https://lmstudio.ai/static/media/demo2.9df5a0e5a9f1d72715e0.gif" width=40%>



## Javascript!

You can write Javascript now to your heart's content! Examples of how this works after a short preamble, or you can scroll straight to them below.

### Using JS fetch method calling LLM

security issue, but u can consider as follows

  * https://github.com/pmcculler/sd-dynamic-javascript
  * https://github.com/ThereforeGames/unprompted
  * https://github.com/adieyal/sd-dynamic-prompts
  * https://en.wikipedia.org/wiki/Server-side_request_forgery
  * and Command Line Arg --allow-code

## Colophon

Made for fun. I hope if brings you great joy, and perfect hair forever. Contact me with questions and comments, but not threats, please. And feel free to contribute! Pull requests and ideas in Discussions or Issues will be taken quite seriously!
--- https://decade.tw

