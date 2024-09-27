
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
let currentRequestID = '';
let idle = true;
let inserted_image = null;


function disableGenerateButton() {
    document.getElementById('send-btn').disabled = true;
};

function enableGenerateButton() {
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').innerHTML = '<i class="fa fa-paper-plane"></i>';
    document.getElementById('send-btn').disabled = false;
    idle = true;
};

function enableAbortButton() {
    if (!idle) {
        return;
    }
    document.getElementById('send-btn').style.backgroundColor = "#363d46";
    document.getElementById('send-btn').innerHTML = '<i class="fa fa-stop"></i>';
    document.getElementById('send-btn').disabled = false;
    idle = false;
}

function updateBotMessage(message, replace = false) {
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
    if (replace) {
        botMessage.innerHTML = message;
    }
    else {
        botMessage.innerHTML += message;
    }
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
            disableGenerateButton();
            return;
        case "finished":
            enableGenerateButton();
            return;
        case "waiting":
            queue_pos = response_dict["queue_pos"];
            updateBotMessage("<br>" + "<span style='color:blue;'>You are in position " + queue_pos + " in the queue</span>", replace = true);
            enableAbortButton();
            return;
        case "aborted":
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
    console.log("Sending message: " + prompt);
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
    // Create a separate container for image
    userMessage.textContent = prompt;
    if (inserted_image !== null) {
        let imageContainer = document.createElement('div');
        imageContainer.classList.add('message-images');
        image = inserted_image.cloneNode(true);
        image.style.maxWidth = '500px';
        image.style.height = 'auto';
        image.style.marginRight = '5px';
        image.style.marginBottom = '5px';
        image.style.borderRadius = '5px';
        imageContainer.appendChild(image);
        userMessage.insertBefore(imageContainer, userMessage.firstChild);

        inserted_image = image.src;
    }
    userMessageContainer.appendChild(userMessage);

    document.getElementById('chat-output').insertBefore(userMessageContainer, document.getElementById('chat-output').firstChild);
    console.log (inserted_image);
    ws.send(JSON.stringify({
        "prompt": prompt,
        "reply_prefix": reply_prefix,
        "system_prompt": system_prompt,
        "image": inserted_image,
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
    document.getElementById('image-preview').innerHTML = '';
    inserted_image = null;
}

function sendButtonClick() {
    document.getElementById('send-btn').disabled = true;
    if (idle) {
        sendMessage();
    }
    else {
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
    document.getElementById('send-btn').innerHTML = '<i class="fa fa-paper-plane"></i>';
    document.getElementById('send-btn').disabled = false;
    console.log("Generation aborted.");
}

function enterSend(event) {
    if (event.key === 'Enter' && !event.ctrlKey) {
        event.preventDefault();
        if (idle){
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
document.addEventListener('click', (event) => {
    const target = event.target;
    if (!configContainer.contains(target) && !menuToggle.contains(target)) {
        configContainer.classList.remove('show');
    }
});

document.addEventListener('paste', function (event) {
    var items = (event.clipboardData || event.originalEvent.clipboardData).items;

    for (var index in items) {
        var item = items[index];
        if (item.kind === 'file') {
            var blob = item.getAsFile();
            var reader = new FileReader();
            reader.onload = function (event) {
                var img = document.createElement("img");
                img.src = event.target.result;

                // Insert only one image
                console.log(img.src);
                if (!img.src.startsWith('data:image') || inserted_image !== null) {
                    return;
                }


                img.style.maxWidth = '100px';
                img.style.height = 'auto';
                img.style.marginRight = '5px';
                img.style.marginBottom = '5px';
                img.style.borderRadius = '5px';

                // Add remove button
                var removeBtn = document.createElement("button");
                removeBtn.innerHTML = "×";
                removeBtn.className = "remove-img-btn";
                removeBtn.onclick = function () {
                    this.parentElement.remove();
                    inserted_image = null;
                };

                var imgContainer = document.createElement("div");
                imgContainer.className = "attached-img-container";
                imgContainer.appendChild(img);
                imgContainer.appendChild(removeBtn);
                document.getElementById('image-preview').appendChild(imgContainer);
                inserted_image = img;
            };
            reader.readAsDataURL(blob);
        }
    }
});

document.addEventListener('DOMContentLoaded', function () {
    menuToggle.addEventListener('click', () => {
        configContainer.classList.toggle('show');
    });

    // Add image attachment functionality
    const attachImageBtn = document.getElementById('attach-image');
    const imageInput = document.createElement('input');
    imageInput.type = 'file';
    imageInput.accept = 'image/*';
    imageInput.style.display = 'none';
    document.body.appendChild(imageInput);

    attachImageBtn.addEventListener('click', function() {
        imageInput.click();
    });

    imageInput.addEventListener('change', function(event) {
        console.log(event);
        handleImageSelection(event.target.files[0]);
    });
});

document.addEventListener('click', (event) => {
    const target = event.target;
    if (!configContainer.contains(target) && !menuToggle.contains(target)) {
        configContainer.classList.remove('show');
    }
});

document.addEventListener('paste', function (event) {
    var items = (event.clipboardData || event.originalEvent.clipboardData).items;
    console.log(items);
    for (var index in items) {
        var item = items[index];
        if (item.kind === 'file') {
            var blob = item.getAsFile();
            handleImageSelection(blob);
            break;  // Only handle the first image
        }
    }
});

function handleImageSelection(file) {
    if (file && file.type.startsWith('image/')) {
        var reader = new FileReader();
        reader.onload = function (event) {
            var img = document.createElement("img");
            img.src = event.target.result;

            if (inserted_image !== null) {
                // Remove existing image
                document.getElementById('image-preview').innerHTML = '';
            }

            img.style.maxWidth = '100px';
            img.style.height = 'auto';
            img.style.marginRight = '5px';
            img.style.marginBottom = '5px';
            img.style.borderRadius = '5px';

            // Add remove button
            var removeBtn = document.createElement("button");
            removeBtn.innerHTML = "×";
            removeBtn.className = "remove-img-btn";
            removeBtn.onclick = function () {
                this.parentElement.remove();
                inserted_image = null;
            };

            var imgContainer = document.createElement("div");
            imgContainer.className = "attached-img-container";
            imgContainer.appendChild(img);
            imgContainer.appendChild(removeBtn);
            document.getElementById('image-preview').appendChild(imgContainer);
            inserted_image = img;
        };
        reader.readAsDataURL(file);
    }
}