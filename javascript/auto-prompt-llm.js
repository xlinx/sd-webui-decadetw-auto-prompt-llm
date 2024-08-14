function sendPromptAutoPromptLLM(tabName, prompt){
    console.log("[][sendPromptAutoPromptLLM][]")
    const textArea = gradioApp().querySelector("#" + tabName + "_prompt > label > textarea");
    let trimCurrentPrompt = textArea.value.trimEnd();
    if (trimCurrentPrompt !== '') {
        if (trimCurrentPrompt.endsWith(',')) {
            textArea.value = trimCurrentPrompt + ' ' + prompt;
        } else {
            textArea.value = trimCurrentPrompt + ', ' + prompt;
        }
    } else {
        textArea.value = prompt;
    }
    updateInput(textArea);
    if (tabName === 'txt2img') {
        switch_to_txt2img();
    } else if (tabName === 'img2img') {
        switch_to_img2img();
    }
}