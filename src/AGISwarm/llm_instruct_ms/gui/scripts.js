
let ws = new WebSocket(WEBSOCKET_URL);
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
let currentRequestID = '';


function disableGenerateButton() {
    document.getElementById('send-btn').disabled = true;
};

function enableGenerateButton() {
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').textContent = "Send";
    document.getElementById('send-btn').disabled = false;
};

function enableAbortButton() {
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').textContent = "Abort";
    document.getElementById('send-btn').disabled = false;
}

function updateBotMessage(message) {
    const chatOutput = document.getElementById('chat-output');
    // Check if the bot message div already exists
    let botMessageContainer = chatOutput.firstElementChild;
    let botMessage = botMessageContainer ? botMessageContainer.firstElementChild : null;
    if (!botMessage || !botMessage.classList.contains('bot-message')) {
        botMessageContainer = document.createElement('div');
        botMessageContainer.classList.add('message-container');
        botMessage = document.createElement('pre');
        botMessage.classList.add('message');
        botMessage.classList.add('bot-message');
        botMessageContainer.appendChild(botMessage);
        chatOutput.insertBefore(botMessageContainer, chatOutput.firstChild);
    }
    botMessage.textContent += message;
    botMessage.style.color = 'black';
    const isAtBottom = chatOutput.scrollHeight - chatOutput.clientHeight <= chatOutput.scrollTop + 1;
    if (isAtBottom) {
        // If the user is at the bottom, scroll to the bottom
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }
}


ws.onmessage = function (event) {
    // Send button is disabled until the response is received
    response_dict = JSON.parse(event.data);
    console.log(response_dict);
    currentRequestID = JSON.parse(event.data)["task_id"];
    
    switch (response_dict["status"]) {
        case "starting":
            currentMessage = '';
            disableGenerateButton();
            return;
        case "finished":
            currentMessage = '';
            enableGenerateButton();
            return;
        case "waiting":
            queue_pos = response_dict["queue_pos"];
            botMessage.textContent = "Waiting in queue. Position: " + queue_pos;
            botMessage.style.color = 'blue';
            enableAbortButton();
            return;
        case "abort":
            updateBotMessage("<br>" + "<span style='color:red;'>Generation aborted</span>");
            enableAbortButton();
            return;
        case "error":
            updateBotMessage("<br>" + "<span style='color:red;'>Error in generation</span>");
            enableGenerateButton();
            return;
        case "running":
            updateBotMessage(response_dict["tokens"]);
            enableAbortButton();
            return;
    }
};


ws.onclose = function (event) {
    console.log("WebSocket is closed now.");
};
function sendMessage() {
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
        systemMessageContainer.classList.add('message-container');
        system_message = document.createElement('pre');
        system_message.classList.add('message');
        system_message.classList.add('system-message');
        system_message.textContent = system_prompt;
        systemMessageContainer.appendChild(system_message);
        document.getElementById('chat-output').insertBefore(systemMessageContainer, document.getElementById('chat-output').firstChild);
    }
    userMessageContainer = document.createElement('div');
    userMessageContainer.classList.add('message-container');
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

function sendButtonClick() {
    document.getElementById('send-btn').disabled = true;
    if (document.getElementById('send-btn').textContent === "Send") {
        sendMessage();
    }
    else if (document.getElementById('send-btn').textContent === "Abort") {
        abortGeneration();
    }
}

function abortGeneration() {
    console.log(currentRequestID)
    fetch(ABORT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'task_id': currentRequestID })
    })
        .then(response => response.text())
        .catch(error => console.error('Error aborting generation:', error));
    // Enable the send button
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').textContent = "Send";
    document.getElementById('send-btn').disabled = false;
    console.log("Generation aborted.");
}

function enterSend(event) {
    if (event.key === 'Enter' && !event.ctrlKey) {
        event.preventDefault();
        if (document.getElementById('send-btn').textContent === "Send") {
            document.getElementById('send-btn').disabled = true;
            sendMessage();
        }
    } else if (event.key === 'Enter' && event.ctrlKey) {
        // Allow new line with Ctrl+Enter
        this.value += '\n';
    }
}

document.getElementById('prompt').addEventListener('keydown', enterSend);
document.getElementById('reply_prefix').addEventListener('keydown', enterSend);
document.getElementById('system_prompt').addEventListener('keydown', enterSend);

const menuToggle = document.getElementById('menu-toggle');
const configContainer = document.querySelector('.config-container');
document.addEventListener('DOMContentLoaded', function () {

    menuToggle.addEventListener('click', () => {
        configContainer.classList.toggle('show');
    });

});
document.addEventListener('click', (event) => {
    const target = event.target;
    if (!configContainer.contains(target) && !menuToggle.contains(target)) {
        configContainer.classList.remove('show');
    }
});