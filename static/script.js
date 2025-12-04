document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const sendBtn = document.getElementById('send-btn');
    const compareBtn = document.getElementById('compare-btn');
    const chatMessages = document.getElementById('chat-messages');
    const userMessage = document.getElementById('user-message');
    const chatDisplay = document.getElementById('chat-display');
    const memoryDisplay = document.getElementById('memory-display');
    const personalitySelect = document.getElementById('personality-select');
    const compareDisplay = document.getElementById('compare-display');
    
    let memoryData = null;

    // Format memory data for display
    function formatMemoryForDisplay(memory) {
        if (!memory) return 'No memory data available';
        
        let html = '';

        // Preferences: backend returns a dict of category -> array of preference objects
        if (memory.preferences && Object.keys(memory.preferences).length > 0) {
            html += '<div class="font-semibold text-blue-600">Preferences:</div>';
            html += '<ul class="list-disc pl-5 mb-2">';
            for (const [category, prefs] of Object.entries(memory.preferences)) {
                if (Array.isArray(prefs) && prefs.length > 0) {
                    prefs.forEach(pref => {
                        const value = pref.value ?? '';
                        const confidence = typeof pref.confidence === 'number'
                            ? ` (confidence: ${(pref.confidence * 100).toFixed(0)}%)`
                            : '';
                        const context = pref.context ? ` – <span class="text-gray-500">${pref.context}</span>` : '';
                        html += `<li><span class="font-medium">${category}:</span> ${value}${confidence}${context}</li>`;
                    });
                }
            }
            html += '</ul>';
        }
        
        // Emotional patterns: array of objects
        if (Array.isArray(memory.emotional_patterns) && memory.emotional_patterns.length > 0) {
            html += '<div class="font-semibold text-purple-600 mt-2">Emotional Patterns:</div>';
            html += '<ul class="list-disc pl-5 mb-2">';
            memory.emotional_patterns.forEach(p => {
                const pattern = p.pattern ?? '';
                const freq = typeof p.frequency === 'number'
                    ? ` (frequency: ${(p.frequency * 100).toFixed(0)}%)`
                    : '';
                const intensity = p.intensity ? ` – intensity: ${p.intensity}` : '';
                const context = p.context ? ` – <span class="text-gray-500">${p.context}</span>` : '';
                html += `<li>${pattern}${freq}${intensity}${context}</li>`;
            });
            html += '</ul>';
        }
        
        // Facts: array of objects
        if (Array.isArray(memory.facts) && memory.facts.length > 0) {
            html += '<div class="font-semibold text-green-600 mt-2">Facts:</div>';
            html += '<ul class="list-disc pl-5">';
            memory.facts.forEach(f => {
                const factType = f.fact_type ?? 'fact';
                const value = f.value ?? '';
                const confidence = typeof f.confidence === 'number'
                    ? ` (confidence: ${(f.confidence * 100).toFixed(0)}%)`
                    : '';
                const temporal = f.temporal_relevance ? ` – ${f.temporal_relevance}` : '';
                const context = f.source_context ? ` – <span class="text-gray-500">${f.source_context}</span>` : '';
                html += `<li><span class="font-medium">${factType}:</span> ${value}${confidence}${temporal}${context}</li>`;
            });
            html += '</ul>';
        }
        
        return html || 'No memory data available';
    }

    // Add message to chat display
    function addMessage(role, content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
        
        if (!isUser) {
            const personality = personalitySelect.options[personalitySelect.selectedIndex].text;
            messageDiv.innerHTML = `
                <div class="personality-tag">${personality}</div>
                <div>${content.replace(/\n/g, '<br>')}</div>
            `;
        } else {
            messageDiv.textContent = content;
        }
        
        chatDisplay.prepend(messageDiv);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    // Analyze messages
    analyzeBtn.addEventListener('click', async function() {
        const messages = chatMessages.value.trim();
        if (!messages) {
            alert('Please enter some chat messages to analyze');
            return;
        }
        
        // Split messages by newline and filter out empty lines
        const messageList = messages.split('\n')
            .filter(msg => msg.trim() !== '')
            .map(msg => ({
                role: 'user',
                content: msg.replace(/^user:\s*/i, '').trim()
            }));
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Backend expects: { messages: [...], user_id?: string }
                body: JSON.stringify({
                    messages: messageList.slice(0, 30)
                })
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            memoryData = data.memory;
            
            // Update memory display
            memoryDisplay.innerHTML = formatMemoryForDisplay(memoryData);
            
            // Enable send & compare buttons
            sendBtn.disabled = false;
            compareBtn.disabled = false;
            
            // Show success message
            addMessage('ai', 'Analysis complete! Memory has been extracted. Try sending a message to see the personality in action!');
            
        } catch (error) {
            console.error('Error analyzing messages:', error);
            alert('Error analyzing messages. Please check the console for details.');
        }
    });

    // Send message
    async function sendMessage() {
        const message = userMessage.value.trim();
        if (!message || !memoryData) return;
        
        // Add user message to chat
        addMessage('user', message, true);
        userMessage.value = '';
        
        try {
            const personality = personalitySelect.value;
            
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    personality: personality,
                    memory: memoryData
                })
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            addMessage('ai', data.response);
            
        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('ai', 'Sorry, I encountered an error processing your message.');
        }
    }

    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);
    
    // Send message on Enter key
    userMessage.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Handle personality change
    personalitySelect.addEventListener('change', function() {
        if (chatDisplay.children.length > 0) {
            addMessage('ai', `Personality changed to: ${this.options[this.selectedIndex].text}`);
        }
    });

    // Compare personalities using the existing /api/compare endpoint
    async function comparePersonalities() {
        const message = userMessage.value.trim();
        if (!message) {
            alert('Please enter a message to compare personalities.');
            return;
        }

        // For the demo, compare a fixed set of personalities
        const personalities = ['mentor', 'friend', 'therapist', 'professional'];

        compareDisplay.innerHTML = '<p class="text-gray-500 italic">Comparing personalities...</p>';

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    personalities: personalities
                })
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const data = await response.json();

            // Build comparison view
            let html = '';
            html += '<div class="mb-3">';
            html += '<div class="font-semibold text-gray-800 mb-1">User message:</div>';
            html += `<div class="mb-2">${data.user_message || message}</div>`;
            html += '<div class="font-semibold text-gray-800 mb-1">Base response:</div>';
            html += `<div class="bg-white border rounded p-2 mb-3">${(data.base_response || '').replace(/\n/g, '<br>')}</div>`;
            html += '</div>';

            if (data.personality_responses) {
                html += '<div class="font-semibold text-gray-800 mb-1">Personality responses:</div>';
                html += '<div class="grid md:grid-cols-2 gap-3 mb-3">';
                Object.entries(data.personality_responses).forEach(([ptype, text]) => {
                    html += '<div class="bg-white border rounded p-2">';
                    html += `<div class="text-sm font-semibold text-purple-700 mb-1">${ptype}</div>`;
                    html += `<div>${String(text).replace(/\n/g, '<br>')}</div>`;
                    html += '</div>';
                });
                html += '</div>';
            }

            if (data.comparison_analysis) {
                html += '<div class="font-semibold text-gray-800 mb-1">Comparison analysis:</div>';
                html += '<pre class="whitespace-pre-wrap text-xs bg-white border rounded p-2 mb-2">'
                    + `${JSON.stringify(data.comparison_analysis, null, 2)}` + '</pre>';
            }

            if (Array.isArray(data.recommendations) && data.recommendations.length > 0) {
                html += '<div class="font-semibold text-gray-800 mb-1">Recommendations:</div>';
                html += '<ul class="list-disc pl-5 text-xs">';
                data.recommendations.forEach(r => {
                    html += `<li>${r}</li>`;
                });
                html += '</ul>';
            }

            compareDisplay.innerHTML = html || '<p class="text-gray-500 italic">No comparison data available.</p>';

        } catch (error) {
            console.error('Error comparing personalities:', error);
            compareDisplay.innerHTML = '<p class="text-red-500 text-sm">Error comparing personalities. Please check the console for details.</p>';
        }
    }

    // Compare button click
    if (compareBtn) {
        compareBtn.addEventListener('click', comparePersonalities);
    }
});
