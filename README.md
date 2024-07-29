# sd-webui-decadetw-auto-prompt-llm
* Automatic1111 extension 
* Calling LLM; auto prompt for batch generate images 
* embedding LLM into prompts.
* embedding Javascript into prompts.

## Motivation

* batch image generate with LLM
* prompt detail is important
* detail using LLM
* idea form https://github.com/pmcculler/sd-dynamic-javascript
* Currently you can't embed Javascript in your SD prompts, which is just silly.
* That's sufficent, I think. I can't wait to see what amazing things people come up with. Please share them with me, and others, if you pease.



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

![readme0.png](images/readme0.png)

![readme1.png](images/readme1.png)

## Installtion

### Suggestion Model

* 4B VRAM<2G
  * CHE-72/Qwen1.5-4B-Chat-Q2_K-GGUF/qwen1.5-4b-chat-q2_k.gguf
    * https://huggingface.co/CHE-72/Qwen1.5-4B-Chat-Q2_K-GGUF
* 7B VRAM<8G
  * ccpl17/Llama-3-Taiwan-8B-Instruct-GGUF/Llama-3-Taiwan-8B-Instruct.Q2_K.gguf
  * Lewdiculous/L3-8B-Stheno-v3.2-GGUF-IQ-Imatrix/L3-8B-Stheno-v3.2-IQ3_XXS-imat.gguf

### Suggestion software


* https://lmstudio.ai/
* https://github.com/openai/openai-python

<img src="https://lmstudio.ai/static/media/demo2.9df5a0e5a9f1d72715e0.gif" width=40%>



## Javascript!

You can write Javascript now to your heart's content! Examples of how this works after a short preamble, or you can scroll straight to them below.



### Using JS fetch method calling LLM

```javascript
%%
 result = await fetch('http://localhost:1234/v1/chat/completions', {
            method: 'POST',
            body: JSON.stringify({
                "messages": [{
                    "role": "system",
                    "content": "You are an AI prompt word playwright. Use the provided keywords to create a beautiful composition. You only need the prompt words, not your feelings. Customize the style, pose, lighting, scene, decoration, etc., and be as detailed as possible. "
                }, {"role": "user", "content": "A superstar on sex."}],
                "temperature": 0.7,
                "max_tokens": 80,
                "stream": false
            }),
            headers: {'Content-Type': 'application/json'}
        }).then(res => {
            return res.json();
        }).then(result => {
            return result['choices'][0]['message']['content'];
        });

        return result;
%%
```



## Colophon

Made for fun. I hope if brings you great joy, and perfect hair forever. Contact me with questions and comments, but not threats, please. And feel free to contribute! Pull requests and ideas in Discussions or Issues will be taken quite seriously!
--- https://decade.tw

