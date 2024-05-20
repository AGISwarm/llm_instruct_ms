
let ws = new WebSocket(WEBSOCKET_URL); // "wss://8fb7-188-88-138-217.ngrok-free.app/ws"
ws.onopen = function () {
    console.log("WebSocket is open now.");
};

function resetForm() {
    document.getElementById('max_new_tokens').value = DEFAULT_MAX_NEW_TOKENS;
    document.getElementById('temperature').value = DEFAULT_TEMPERATURE;
    document.getElementById('top_p').value = DEFAULT_TOP_P;
    document.getElementById('repetition_penalty').value = DEFAULT_REPETITION_PENALTY;
    document.getElementById('frequency_penalty').value = DEFAULT_FREQUENCY_PENALTY;
    document.getElementById('presence_penalty').value = DEFAULT_PRESENCE_PENALTY;
    document.getElementById('system_prompt').value = DEFAULT_SYSTEM_PROMPT;
}
let currentMessage = '';
ws.onmessage = function (event) {
    // Send button is disabled until the response is received
    const token = event.data;
    if (token == "<end_of_response>") {
        currentMessage = '';
        document.getElementById('send-btn').disabled = false;
        return;
    }
    currentMessage += token;
    const chatOutput = document.getElementById('chat-output');
    // Check if the bot message div already exists
    let botMessageContainer = chatOutput.firstElementChild;
    let botMessage = botMessageContainer ? botMessageContainer.firstElementChild : null;
    if (!botMessage || !botMessage.classList.contains('bot-message')) {
        botMessageContainer = document.createElement('div');
        botMessage = document.createElement('pre');
        botMessage.classList.add('message');
        botMessage.classList.add('bot-message');
        botMessageContainer.appendChild(botMessage);
        chatOutput.insertBefore(botMessageContainer, chatOutput.firstChild);
    }
    botMessage.textContent = currentMessage;
    const isAtBottom = chatOutput.scrollHeight - chatOutput.clientHeight <= chatOutput.scrollTop + 1;
    if (isAtBottom) {
        // If the user is at the bottom, scroll to the bottom
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }
};
ws.onclose = function (event) {
    console.log("WebSocket is closed now.");
};
function sendMessage() {
    document.getElementById('send-btn').disabled = true;
    const prompt = document.getElementById('prompt').value;
    const max_new_tokens = document.getElementById('max_new_tokens').value;
    const reply_prefix = document.getElementById('reply_prefix').value;
    const system_prompt = document.getElementById('system_prompt').value;
    const temperature = document.getElementById('temperature').value;
    const top_p = document.getElementById('top_p').value;
    const repetition_penalty = document.getElementById('repetition_penalty').value;
    const frequency_penalty = document.getElementById('frequency_penalty').value;
    const presence_penalty = document.getElementById('presence_penalty').value;

    if (system_prompt != '') {
        systemMessageContainer = document.createElement('div');
        system_message = document.createElement('pre');
        system_message.classList.add('message');
        system_message.classList.add('system-message');
        system_message.textContent = system_prompt;
        systemMessageContainer.appendChild(system_message);
        document.getElementById('chat-output').insertBefore(systemMessageContainer, document.getElementById('chat-output').firstChild);
    }
    userMessageContainer = document.createElement('div');
    userMessage = document.createElement('pre');
    userMessage.classList.add('message');
    userMessage.classList.add('user-message');
    userMessage.textContent = prompt;
    userMessageContainer.appendChild(userMessage);
    document.getElementById('chat-output').insertBefore(userMessageContainer, document.getElementById('chat-output').firstChild);
    ws.send(JSON.stringify({
        "prompt": prompt,
        "reply_prefix": reply_prefix,
        "system_prompt": system_prompt,
        "max_new_tokens": parseInt(max_new_tokens),
        "temperature": parseFloat(temperature),
        "top_p": parseFloat(top_p),
        "repetition_penalty": parseFloat(repetition_penalty),
        "frequency_penalty": parseFloat(frequency_penalty),
        "presence_penalty": parseFloat(presence_penalty)
    }));
    document.getElementById('prompt').value = '';
    document.getElementById('system_prompt').value = '';
    document.getElementById('reply_prefix').value = '';
}
document.getElementById('prompt').addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.ctrlKey) {
        event.preventDefault();
        sendMessage();
    } else if (event.key === 'Enter' && event.ctrlKey) {
        // Allow new line with Ctrl+Enter
        this.value += '\n';
    }
});