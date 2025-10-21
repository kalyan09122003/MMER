// EmotiAI Frontend JavaScript
// Handles all interactions for text, audio, image, and video

// API Configuration - Always use the same origin as the frontend
const API_BASE_URL = window.location.origin;

// Global variables
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let videoStream = null;
let isVideoAnalyzing = false;
let videoAnalysisInterval = null;
let emotionHistory = []; // For temporal smoothing
let maxHistoryLength = 5;

// Utility Functions
function showElement(element) {
    if (element) {
        element.style.display = 'block';
        element.classList.remove('hidden');
    }
}

function hideElement(element) {
    if (element) {
        element.style.display = 'none';
        element.classList.add('hidden');
    }
}

function updateProgressBar(emotion, percentage) {
    const progressElement = document.getElementById(`${emotion}Progress`);
    const percentageElement = document.getElementById(`${emotion}Percentage`);
    
    if (progressElement && percentageElement) {
        progressElement.style.width = `${percentage}%`;
        percentageElement.textContent = `${Math.round(percentage)}%`;
    }
    
    // Debug logging
    console.log(`Updating ${emotion}: ${percentage}%`, {
        progressElement: !!progressElement,
        percentageElement: !!percentageElement
    });
}

function displayEmotionResults(data, pageType = 'general') {
    console.log('Displaying emotion results:', data, 'Page type:', pageType);
    
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const emotionResults = document.getElementById('emotionResults');
    const dominantEmotion = document.getElementById('dominantEmotion');
    const confidenceScore = document.getElementById('confidenceScore');
    
    console.log('Elements found:', {
        resultsPlaceholder: !!resultsPlaceholder,
        emotionResults: !!emotionResults,
        dominantEmotion: !!dominantEmotion,
        confidenceScore: !!confidenceScore
    });
    
    if (resultsPlaceholder) {
        hideElement(resultsPlaceholder);
        console.log('Hiding results placeholder');
    }
    if (emotionResults) {
        showElement(emotionResults);
        console.log('Showing emotion results section');
    } else {
        console.error('emotionResults element not found!');
    }
    
    // Update dominant emotion
    if (dominantEmotion && data.label) {
        dominantEmotion.textContent = data.label.charAt(0).toUpperCase() + data.label.slice(1);
    }
    
    // Calculate confidence (highest probability)
    let maxProb = 0;
    if (data.probabilities) {
        maxProb = Math.max(...Object.values(data.probabilities)) * 100;
    }
    
    if (confidenceScore) {
        confidenceScore.textContent = `Confidence: ${Math.round(maxProb)}%`;
    }
    
    // Update progress bars based on page type
    if (data.probabilities) {
        if (pageType === 'image' || pageType === 'video') {
            // Image/Video uses FER emotions
            const emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'];
            emotions.forEach(emotion => {
                const prob = (data.probabilities[emotion] || 0) * 100;
                updateProgressBar(emotion, prob);
            });
        } else {
            // Text/Audio uses text-based emotions
            const emotions = ['joy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'];
            emotions.forEach(emotion => {
                const prob = (data.probabilities[emotion] || 0) * 100;
                updateProgressBar(emotion, prob);
            });
        }
    }
    
    // Update faces count for image/video
    if (pageType === 'image' || pageType === 'video') {
        const facesCount = document.getElementById('facesCount');
        // Use data.faces if available, otherwise fallback to 1 if probabilities exist
        if (facesCount) {
            if (data.faces !== undefined) {
                facesCount.textContent = data.faces;
            } else if (data.probabilities) {
                facesCount.textContent = 1;
            } else {
                facesCount.textContent = 0;
            }
        }
    }
}

// Temporal smoothing function for video analysis
function temporalSmoothing(currentEmotion, alpha = 0.7) {
    if (emotionHistory.length === 0) {
        emotionHistory.push(currentEmotion);
        return currentEmotion;
    }
    
    const lastEmotion = emotionHistory[emotionHistory.length - 1];
    const smoothedEmotion = {};
    
    // Smooth each emotion probability
    for (const emotion in currentEmotion) {
        if (lastEmotion[emotion] !== undefined) {
            smoothedEmotion[emotion] = alpha * currentEmotion[emotion] + (1 - alpha) * lastEmotion[emotion];
        } else {
            smoothedEmotion[emotion] = currentEmotion[emotion];
        }
    }
    
    // Add to history and maintain max length
    emotionHistory.push(smoothedEmotion);
    if (emotionHistory.length > maxHistoryLength) {
        emotionHistory.shift();
    }
    
    return smoothedEmotion;
}

// Confidence-based display filtering
function shouldDisplayResult(data, minConfidence = 0.4) {
    if (data.confidence !== undefined) {
        return data.confidence >= minConfidence;
    }
    
    // For text/audio, check if dominant emotion probability is high enough
    if (data.probabilities) {
        const maxProb = Math.max(...Object.values(data.probabilities));
        return maxProb >= minConfidence;
    }
    
    return true; // Default to showing result
}

function showError(message) {
    alert(`Error: ${message}`);
    console.error('EmotiAI Error:', message);
}

// Text Analysis Functions
function initTextAnalysis() {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const sampleItems = document.querySelectorAll('.sample-item');
    
    if (!textInput) return; // Not on text page
    
    // Character counter
    textInput.addEventListener('input', () => {
        if (charCount) {
            charCount.textContent = textInput.value.length;
        }
    });
    
    // Analyze button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeText);
    }
    
    // Sample text items
    sampleItems.forEach(item => {
        item.addEventListener('click', () => {
            const sampleText = item.getAttribute('data-text');
            if (sampleText && textInput) {
                textInput.value = sampleText;
                if (charCount) {
                    charCount.textContent = sampleText.length;
                }
            }
        });
    });
}

// Text analysis using the real API
async function analyzeTextWithAPI(text) {
    if (!API_BASE_URL) {
        throw new Error('API base URL is not configured');
    }
    
    const response = await fetch(`${API_BASE_URL}/predict_text`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

async function analyzeText() {
    const textInput = document.getElementById('textInput');
    const analyzeText = document.getElementById('analyzeText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    if (!textInput || !textInput.value.trim()) {
        showError('Please enter some text to analyze');
        return;
    }
    
    // Show loading state
    if (analyzeText) hideElement(analyzeText);
    if (loadingSpinner) showElement(loadingSpinner);
    
    try {
        const data = await analyzeTextWithAPI(textInput.value.trim());
        console.log('Analysis data:', data);
        displayEmotionResults(data, 'text');
    } catch (error) {
        console.error('Error analyzing text:', error);
        showError(`Failed to analyze text: ${error.message}`);
    } finally {
        // Hide loading state
        if (loadingSpinner) hideElement(loadingSpinner);
        if (analyzeText) showElement(analyzeText);
    }
}

// Audio Analysis Functions
function initAudioAnalysis() {
    const micButton = document.getElementById('micButton');
    
    if (!micButton) return; // Not on audio page
    
    micButton.addEventListener('click', toggleRecording);
    
    // Initialize audio visualizer
    initAudioVisualizer();
}

function initAudioVisualizer() {
    const canvas = document.getElementById('audioVisualizer');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Draw static visualizer when not recording
    function drawStaticVisualizer() {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = 'rgba(79, 195, 247, 0.3)';
        
        for (let i = 0; i < 20; i++) {
            const barHeight = Math.random() * 30 + 10;
            const x = (i * width) / 20;
            ctx.fillRect(x, height - barHeight, width / 25, barHeight);
        }
    }
    
    drawStaticVisualizer();
}

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    const micButton = document.getElementById('micButton');
    const micIcon = document.getElementById('micIcon');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingInstruction = document.getElementById('recordingInstruction');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await analyzeAudio(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        if (micButton) micButton.classList.add('recording');
        if (micIcon) micIcon.textContent = '⏹️';
        if (recordingStatus) recordingStatus.textContent = 'Recording...';
        if (recordingInstruction) recordingInstruction.textContent = 'Click to stop recording';
        
    } catch (error) {
        showError(`Failed to access microphone: ${error.message}`);
    }
}

function stopRecording() {
    const micButton = document.getElementById('micButton');
    const micIcon = document.getElementById('micIcon');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingInstruction = document.getElementById('recordingInstruction');
    
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        
        // Update UI
        if (micButton) micButton.classList.remove('recording');
        if (micIcon) micIcon.textContent = '🎤';
        if (recordingStatus) recordingStatus.textContent = 'Processing...';
        if (recordingInstruction) recordingInstruction.textContent = 'Analyzing audio...';
    }
}

async function analyzeAudio(audioBlob) {
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingInstruction = document.getElementById('recordingInstruction');
    const transcriptionSection = document.getElementById('transcriptionSection');
    const transcriptionText = document.getElementById('transcriptionText');
    
    try {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');
        
        const response = await fetch(`${API_BASE_URL}/predict_audio`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display transcription
        if (transcriptionSection && transcriptionText && data.text) {
            transcriptionText.textContent = data.text || 'No speech detected';
            showElement(transcriptionSection);
        }
        
        // Display emotion results
        displayEmotionResults(data, 'audio');
        
        // Update UI
        if (recordingStatus) recordingStatus.textContent = 'Analysis Complete';
        if (recordingInstruction) recordingInstruction.textContent = 'Click the microphone to record again';
        
    } catch (error) {
        showError(`Failed to analyze audio: ${error.message}`);
        if (recordingStatus) recordingStatus.textContent = 'Analysis Failed';
        if (recordingInstruction) recordingInstruction.textContent = 'Click the microphone to try again';
    }
}

// Image Analysis Functions
function initImageAnalysis() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const analyzeImageBtn = document.getElementById('analyzeImageBtn');
    
    if (!uploadArea) return; // Not on image page
    
    // Upload area click
    uploadArea.addEventListener('click', () => {
        if (imageInput) imageInput.click();
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    });
    
    // File input change
    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageFile(e.target.files[0]);
            }
        });
    }
    
    // Analyze button
    if (analyzeImageBtn) {
        analyzeImageBtn.addEventListener('click', analyzeCurrentImage);
    }
}

function handleImageFile(file) {
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        showError('Image file too large. Please select a file under 10MB');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImg) {
            previewImg.src = e.target.result;
            previewImg.dataset.file = file.name;
        }
        if (imagePreview) {
            showElement(imagePreview);
        }
    };
    reader.readAsDataURL(file);
    
    // Store file for analysis
    window.currentImageFile = file;
}

async function analyzeCurrentImage() {
    const analyzeImageText = document.getElementById('analyzeImageText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    if (!window.currentImageFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading state
    if (analyzeImageText) hideElement(analyzeImageText);
    if (loadingSpinner) showElement(loadingSpinner);
    
    try {
        const formData = new FormData();
        formData.append('image', window.currentImageFile);
        
        const response = await fetch(`${API_BASE_URL}/predict_image`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayEmotionResults(data, 'image');
        
    } catch (error) {
        showError(`Failed to analyze image: ${error.message}`);
    } finally {
        // Hide loading state
        if (loadingSpinner) hideElement(loadingSpinner);
        if (analyzeImageText) showElement(analyzeImageText);
    }
}

// Video Analysis Functions
function initVideoAnalysis() {
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    
    if (!startCameraBtn) return; // Not on video page
    
    startCameraBtn.addEventListener('click', startCamera);
    if (stopCameraBtn) {
        stopCameraBtn.addEventListener('click', stopCamera);
    }
}

async function startCamera() {
    const videoFeed = document.getElementById('videoFeed');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const liveIndicator = document.getElementById('liveIndicator');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const analysisStatus = document.getElementById('analysisStatus');
    const analysisInsights = document.getElementById('analysisInsights');
    
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        if (videoFeed) {
            videoFeed.srcObject = videoStream;
            showElement(videoFeed);
        }
        if (videoPlaceholder) hideElement(videoPlaceholder);
        if (liveIndicator) showElement(liveIndicator);
        if (startCameraBtn) hideElement(startCameraBtn);
        if (stopCameraBtn) showElement(stopCameraBtn);
        if (analysisStatus) {
            analysisStatus.textContent = 'Active';
            analysisStatus.style.color = '#4fc3f7';
        }
        if (analysisInsights) {
            analysisInsights.textContent = 'Camera started. Real-time analysis beginning...';
        }
        
        // Start video analysis
        startVideoAnalysis();
        
    } catch (error) {
        showError(`Failed to access camera: ${error.message}`);
    }
}

function stopCamera() {
    const videoFeed = document.getElementById('videoFeed');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const liveIndicator = document.getElementById('liveIndicator');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const analysisStatus = document.getElementById('analysisStatus');
    const analysisInsights = document.getElementById('analysisInsights');
    
    // Stop video stream
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    // Stop analysis
    if (videoAnalysisInterval) {
        clearInterval(videoAnalysisInterval);
        videoAnalysisInterval = null;
    }
    isVideoAnalyzing = false;
    
    // Update UI
    if (videoFeed) {
        hideElement(videoFeed);
        videoFeed.srcObject = null;
    }
    if (videoPlaceholder) showElement(videoPlaceholder);
    if (liveIndicator) hideElement(liveIndicator);
    if (startCameraBtn) showElement(startCameraBtn);
    if (stopCameraBtn) hideElement(stopCameraBtn);
    if (analysisStatus) {
        analysisStatus.textContent = 'Inactive';
        analysisStatus.style.color = '#666';
    }
    if (analysisInsights) {
        analysisInsights.textContent = 'Camera stopped. Click "Start Camera" to begin analysis.';
    }
}

function startVideoAnalysis() {
    const videoFeed = document.getElementById('videoFeed');
    if (!videoFeed) return;
    
    isVideoAnalyzing = true;
    
    // Capture and analyze frames every 2 seconds
    videoAnalysisInterval = setInterval(async () => {
        if (!isVideoAnalyzing || !videoFeed.srcObject) return;
        
        try {
            // Capture frame from video
            const canvas = document.createElement('canvas');
            canvas.width = videoFeed.videoWidth || 640;
            canvas.height = videoFeed.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
            
            // Convert to blob
            canvas.toBlob(async (blob) => {
                if (blob && isVideoAnalyzing) {
                    await analyzeVideoFrame(blob);
                }
            }, 'image/jpeg', 0.8);
            
        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }, 2000);
}

async function analyzeVideoFrame(frameBlob) {
    const analysisInsights = document.getElementById('analysisInsights');
    
    try {
        const formData = new FormData();
        formData.append('image', frameBlob, 'frame.jpg');
        
        const response = await fetch(`${API_BASE_URL}/predict_image`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) return; // Silently fail for video frames
        
        const data = await response.json();
        
        // Apply temporal smoothing for video analysis
        if (data.probabilities && Object.keys(data.probabilities).length > 0) {
            const smoothedProbabilities = temporalSmoothing(data.probabilities);
            const smoothedLabel = Object.keys(smoothedProbabilities).reduce((a, b) => 
                smoothedProbabilities[a] > smoothedProbabilities[b] ? a : b
            );
            
            // Create smoothed data object
            const smoothedData = {
                ...data,
                probabilities: smoothedProbabilities,
                label: smoothedLabel
            };
            
            // Only display if confidence is sufficient
            if (shouldDisplayResult(smoothedData, 0.3)) {
                displayEmotionResults(smoothedData, 'video');
            }
        } else {
            displayEmotionResults(data, 'video');
        }
        
        // Update insights
        if (analysisInsights) {
            const timestamp = new Date().toLocaleTimeString();
            if (data.faces > 0) {
                const confidenceText = data.confidence ? ` (${Math.round(data.confidence * 100)}% confidence)` : '';
                analysisInsights.textContent = `${timestamp}: Detected ${data.faces} face(s). Dominant emotion: ${data.label}${confidenceText}`;
            } else {
                analysisInsights.textContent = `${timestamp}: No faces detected in current frame.`;
            }
        }
        
    } catch (error) {
        // Silently handle errors for video analysis
        console.error('Video frame analysis error:', error);
    }
}

// Mobile Menu Toggle
function initMobileMenu() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuToggle.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-container') && navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
                menuToggle.classList.remove('active');
                document.body.classList.remove('menu-open');
            }
        });
        
        // Close menu when clicking a nav link
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                menuToggle.classList.remove('active');
                document.body.classList.remove('menu-open');
            });
        });
    }
}

// Initialize based on current page
document.addEventListener('DOMContentLoaded', () => {
    // Initialize mobile menu
    initMobileMenu();
    
    // Initialize based on which page we're on
    initTextAnalysis();
    initAudioAnalysis();
    initImageAnalysis();
    initVideoAnalysis();
    
    console.log('EmotiAI Frontend Initialized');
});
