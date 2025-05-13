// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    const customerNameInput = document.getElementById('customerName');
    const startCallButton = document.getElementById('startCallButton');
    const recordButton = document.getElementById('recordButton');
    const conversationLog = document.getElementById('conversationLog');
    const statusMessage = document.getElementById('statusMessage');
    const botAudioPlayer = document.getElementById('botAudioPlayer');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let callActive = false;

    function addLogMessage(speaker, message) {
        const messageElement = document.createElement('p');
        messageElement.innerHTML = `<strong>${speaker}:</strong> ${message}`;
        conversationLog.appendChild(messageElement);
        conversationLog.scrollTop = conversationLog.scrollHeight; // Auto-scroll
    }

    async function playBotAudio(audioBlob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        botAudioPlayer.src = audioUrl;

        // Ensure previous audio is stopped and player is reset
        botAudioPlayer.currentTime = 0;

        try {
            await botAudioPlayer.play();
            statusMessage.textContent = "Bot is speaking...";
            recordButton.disabled = true; // Disable recording while bot speaks
        } catch (error) {
            console.error("Error playing audio:", error);
            statusMessage.textContent = "Error playing bot response.";
            recordButton.disabled = false; // Re-enable if play fails
        }

        botAudioPlayer.onended = () => {
            URL.revokeObjectURL(audioUrl);
            if (callActive) { // Only re-enable if call is still supposed to be active
                 statusMessage.textContent = "Your turn to speak. Hold/Click record button.";
                 recordButton.disabled = false;
            } else {
                statusMessage.textContent = "Call ended.";
                recordButton.disabled = true;
            }
        };
    }

    startCallButton.addEventListener('click', async () => {
        const customerName = customerNameInput.value.trim();
        // No need to check for customerName here as the bot will ask if not provided.
        // For a better UX, you might want to enforce it on the frontend.

        addLogMessage("System", `Initiating call for: ${customerName || "Unknown Customer"}...`);
        statusMessage.textContent = "Connecting...";
        callActive = true; // Mark call as active

        try {
            const response = await fetch('/initiate_call', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ customerName: customerName })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const audioBlob = await response.blob();
            // The server's response to /initiate_call already contains the first bot message's audio.
            // We need to get the text of this first message to log it.
            // This is a bit of a workaround; ideally, the backend would send text + audio URL or separate them.
            // For now, we'll make a simplified assumption or skip logging the first bot text accurately here.
            // A better approach: /initiate_call returns JSON {text: "...", audio_url: "..."}
            // For this example, we'll log a generic "Bot started."
            addLogMessage("Bot", "Hello, this is LoanMate..."); // Placeholder for first message
            await playBotAudio(audioBlob);
            // recordButton.disabled = false; // Will be enabled after audio plays if call still active

        } catch (error) {
            console.error('Error initiating call:', error);
            addLogMessage("System", `Error: ${error.message}`);
            statusMessage.textContent = "Failed to start call.";
            callActive = false;
        }
    });

    async function startRecording() {
        if (!callActive) {
            statusMessage.textContent = "Please start the call first.";
            return;
        }
        if (isRecording) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                isRecording = false;
                recordButton.textContent = "Hold to Speak (or Click to Start/Stop)";
                recordButton.classList.remove('recording');
                statusMessage.textContent = "Processing your response...";
                recordButton.disabled = true; // Disable while processing

                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // or 'audio/ogg', 'audio/wav'
                audioChunks = []; // Reset for next recording

                const formData = new FormData();
                formData.append('audio_data', audioBlob, 'user_audio.webm');

                try {
                    const response = await fetch('/process_audio', {
                        method: 'POST',
                        body: formData
                    });
                    if (!response.ok) {
                         const errorData = await response.json();
                         throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                    }
                    const newAudioBlob = await response.blob();
                    // The text transcript would ideally come from a JSON response along with audio.
                    // For now, the backend directly sends audio. We need to get the transcript.
                    // This requires the backend to also send the transcript.
                    // Let's assume for now we don't log the user's exact transcript until backend is modified.
                    // addLogMessage("You", "Your spoken response (transcript needed from backend)");
                    // The backend also needs to send the bot's *text* response for logging.
                    // addLogMessage("Bot", "Bot's next textual response (needed from backend)");
                    await playBotAudio(newAudioBlob);

                    // If the bot's response indicates the end of the call, update callActive
                    // This logic needs to be more robust, perhaps by backend sending a 'call_ended' flag
                    // For now, a simple check:
                    // if (bot response implies end) callActive = false;
                    // Example: if (bot_response_text.includes("Have a good day.") || bot_response_text.includes("call ended")) {
                    //    callActive = false;
                    // }


                } catch (error) {
                    console.error('Error processing audio:', error);
                    addLogMessage("System", `Error: ${error.message}`);
                    statusMessage.textContent = "Error processing your response.";
                    recordButton.disabled = false; // Re-enable on error
                }
            };

            mediaRecorder.start();
            isRecording = true;
            recordButton.textContent = "Stop Recording";
            recordButton.classList.add('recording');
            statusMessage.textContent = "Recording... Speak now.";

        } catch (error) {
            console.error('Error accessing microphone:', error);
            addLogMessage("System", "Microphone access denied or error.");
            statusMessage.textContent = "Could not access microphone.";
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            // Stop microphone tracks to release it
            if (mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
    }

    recordButton.addEventListener('click', () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    // Optional: Hold-to-speak functionality (mousedown/mouseup)
    // recordButton.addEventListener('mousedown', startRecording);
    // recordButton.addEventListener('mouseup', stopRecording);
    // For touch devices:
    // recordButton.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
    // recordButton.addEventListener('touchend', (e) => { e.preventDefault(); stopRecording(); });

});