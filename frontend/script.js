const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const followupsSection = document.getElementById('followups');
const contextInfo = document.getElementById('context-info');
const productInfo = document.getElementById('product-info');
const backendStatus = document.getElementById('backend-status');

let chatHistory = [];
const API_URL = 'http://localhost:8000';

// Check backend status
async function checkStatus() {
    try {
        const res = await fetch(`${API_URL}/`);
        if (res.ok) {
            backendStatus.classList.add('online');
            backendStatus.querySelector('span').innerText = 'Backend Online';
        }
    } catch (err) {
        backendStatus.classList.remove('online');
        backendStatus.querySelector('span').innerText = 'Backend Offline';
    }
}

setInterval(checkStatus, 5000);
checkStatus();

async function sendMessage(text) {
    if (!text.trim()) return;

    // Add user message to UI
    appendMessage('user', text);
    userInput.value = '';
    followupsSection.innerHTML = '';

    // Loading state for assistant
    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, history: chatHistory })
        });

        const data = await response.json();
        removeLoadingIndicator(loadingId);

        // Update UI
        appendMessage('assistant', data.answer);
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: data.answer });

        // Update Follow-ups
        if (data.followups && data.followups.length > 0) {
            data.followups.forEach(f => {
                const btn = document.createElement('button');
                btn.className = 'followup-btn';
                btn.innerText = f;
                btn.onclick = () => sendMessage(f);
                followupsSection.appendChild(btn);
            });
        }

        // Update Side Panel
        if (data.context) {
            contextInfo.innerHTML = marked.parse(data.context);
        }
        if (data.products) {
            productInfo.innerHTML = marked.parse(data.products);
        }

    } catch (err) {
        removeLoadingIndicator(loadingId);
        appendMessage('assistant', 'Sorry, I encountered an error connecting to the backend. Please make sure the server is running.');
        console.error(err);
    }
}

function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    // Support markdown in assistant messages
    if (role === 'assistant') {
        div.innerHTML = marked.parse(text);
    } else {
        div.innerText = text;
    }
    
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant loading';
    div.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

chatForm.onsubmit = (e) => {
    e.preventDefault();
    sendMessage(userInput.value);
};

// CSS for typing indicator (added dynamically)
const style = document.createElement('style');
style.textContent = `
    .typing-dot {
        width: 8px;
        height: 8px;
        background: var(--text-gray);
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
        animation: typing 1s infinite alternate;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing {
        from { opacity: 0.3; transform: translateY(0); }
        to { opacity: 1; transform: translateY(-5px); }
    }
`;
document.head.appendChild(style);
