document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');

    // Use sessionStorage to keep session_id only for the browser tab session
    let currentSessionId = sessionStorage.getItem('chatSessionId');

    // --- Function to add message to chatbox ---
    function addMessage(sender, text, type = 'normal') {
        const messageElement = document.createElement('p');
        messageElement.classList.add('message');

        let prefix = '';
        if (sender === 'user') {
            messageElement.classList.add('user');
            prefix = 'You: ';
        } else if (sender === 'ai') {
            messageElement.classList.add('ai');
            prefix = 'AI: ';
        } else if (sender === 'system') { // For system messages/errors
             prefix = ''; // No prefix for system messages, styling handles it
        }

        if (type === 'error') {
            messageElement.classList.add('error');
            prefix = 'Error: ';
        }

        messageElement.textContent = `${prefix}${text}`;
        chatbox.appendChild(messageElement);
        // Scroll to bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    // --- Function to send message to backend ---
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return; // Don't send empty messages

        addMessage('user', messageText); // Display user message immediately
        userInput.value = ''; // Clear input
        sendButton.disabled = true; // Disable button while waiting

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_message: messageText,
                    session_id: currentSessionId // Send null if it's the first message
                }),
            });

            if (!response.ok) {
                // Try to get error details from response body
                let errorText = `API Error: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorText = `API Error: ${errorData.detail || errorText}`;
                } catch (e) { /* Ignore if response body is not JSON */ }
                throw new Error(errorText);
            }

            const data = await response.json();

            addMessage('ai', data.ai_response);

            // IMPORTANT: Store the session ID returned by the backend
            if (data.session_id && !currentSessionId) {
                 currentSessionId = data.session_id;
                 sessionStorage.setItem('chatSessionId', currentSessionId); // Store in sessionStorage
                 console.log("Started new session:", currentSessionId);
            } else if (data.session_id !== currentSessionId) {
                // This case shouldn't usually happen if backend logic is correct, but good to log
                console.warn("Received different session ID from backend:", data.session_id);
                currentSessionId = data.session_id; // Update if needed
                sessionStorage.setItem('chatSessionId', currentSessionId);
            }


        } catch (error) {
            console.error("Failed to send message:", error);
            addMessage('system', `Failed to get response. ${error.message}`, 'error');
        } finally {
             sendButton.disabled = false; // Re-enable button
             userInput.focus(); // Set focus back to input
        }
    }

    // --- Function to clear history --- 
    async function clearHistory() {
        if (!currentSessionId) {
            addMessage('system', "No active session to clear.", 'error');
            return;
        }

        const originalSessionId = currentSessionId; // Keep track in case clear fails
        addMessage('system', "Clearing history...", 'ai'); // Give feedback
        const localSessionIdToClear = currentSessionId; // Store ID to clear
        currentSessionId = null; // Assume clear will work locally first
        sessionStorage.removeItem('chatSessionId'); // Clear from storage

        try {
             const response = await fetch(`/clear_history/${localSessionIdToClear}`, { method: 'POST' });
             if (!response.ok) {
                let errorText = `API Error: ${response.status} ${response.statusText}`;
                try { const errorData = await response.json(); errorText = `API Error: ${errorData.detail || errorText}`; } catch (e) {}
                throw new Error(errorText);
             }
             const data = await response.json();
             console.log("Clear history response:", data.message);
             // Clear chatbox content except the initial AI message and the clear confirmation
             chatbox.innerHTML = '<p class="message ai">AI: Hello! How can I help you today?</p>';
             addMessage('system', "Chat history cleared. Start a new conversation!", 'ai');

        } catch (error) {
             console.error("Failed to clear history on backend:", error);
             addMessage('system', `Failed to clear history on server. ${error.message} You might need to refresh.`, 'error');
             // Restore session ID if backend clear failed
             currentSessionId = originalSessionId;
             sessionStorage.setItem('chatSessionId', currentSessionId);
        }
    }

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        // Send message on Enter key press
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
    clearButton.addEventListener('click', clearHistory);

     // Initial focus
     userInput.focus();

}); // End DOMContentLoaded 