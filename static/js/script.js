// Main script for PlantCare AI app

document.addEventListener('DOMContentLoaded', function() {
    // File input handling
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    const predictBtn = document.getElementById('predictBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const loadingDiv = document.getElementById('loadingDiv');
    const result = document.getElementById('result');
    const uploadForm = document.getElementById('uploadForm');
    const resultImg = document.getElementById('resultImg');

    // Tab functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Handle file selection
    imageInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            
            reader.onload = function(event) {
                previewImg.src = event.target.result;
                imagePreview.style.display = 'block';
                predictBtn.disabled = false;
            };
            
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!imageInput.files.length) {
            alert('Please select an image first.');
            return;
        }
        
        // Show loading animation
        imagePreview.style.display = 'none';
        loadingDiv.style.display = 'block';
        
        // Simulate analysis steps
        setTimeout(() => {
            document.getElementById('step1').classList.remove('active');
            document.getElementById('step2').classList.add('active');
            
            setTimeout(() => {
                document.getElementById('step2').classList.remove('active');
                document.getElementById('step3').classList.add('active');
                
                setTimeout(() => {
                    submitPrediction();
                }, 1000);
            }, 1500);
        }, 1000);
    });

    // Handle tab switching
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Get the tab to show
            const tabToShow = button.getAttribute('data-tab');
            
            // Hide all tab panes
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Show the selected tab pane
            document.getElementById(`${tabToShow}-tab`).classList.add('active');
        });
    });

    // Function to submit image for prediction
    function submitPrediction() {
        const formData = new FormData(uploadForm);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing image. Please try again.');
            loadingDiv.style.display = 'none';
            imagePreview.style.display = 'block';
        });
    }

    // Function to display prediction results
    function displayResults(data) {
        // Hide loading
        loadingDiv.style.display = 'none';
        
        // Update result image
        resultImg.src = previewImg.src;
        
        // Update prediction text
        const predictionText = document.getElementById('predictionText');
        const diseaseClass = data.prediction;
        const confidence = data.confidence;
        
        const healthStatus = diseaseClass.toLowerCase().includes('fresh') ? 'healthy' : 'disease';
        
        predictionText.innerHTML = `
            <div class="prediction-status ${healthStatus}">
                <i class="fas ${healthStatus === 'healthy' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
                <span>${healthStatus === 'healthy' ? 'Healthy' : 'Disease Detected'}</span>
            </div>
            <div class="prediction-name">
                <h4>${diseaseClass}</h4>
                <div class="confidence">
                    <span>Confidence:</span>
                    <span class="confidence-value">${confidence}</span>
                </div>
            </div>
        `;
        
        // Update probabilities list
        const probabilitiesList = document.getElementById('probabilitiesList');
        probabilitiesList.innerHTML = '';
        
        Object.entries(data.all_predictions).forEach(([disease, probability]) => {
            const isHighProbability = parseFloat(probability) > 70;
            const isTopPrediction = disease === diseaseClass;
            
            probabilitiesList.innerHTML += `
                <div class="probability-item ${isTopPrediction ? 'top-prediction' : ''}">
                    <div class="disease-name">
                        ${isTopPrediction ? '<i class="fas fa-star"></i>' : ''}
                        ${disease}
                    </div>
                    <div class="probability-bar-container">
                        <div class="probability-bar" style="width: ${parseFloat(probability)}%"></div>
                        <span class="probability-value">${probability}</span>
                    </div>
                </div>
            `;
        });
        
        // Update model info
        const architectureText = document.getElementById('architectureText');
        architectureText.innerHTML = `
            <p><strong>Architecture:</strong> ${data.model_info.architecture}</p>
            <p><strong>Base Models:</strong> ${data.model_info.base_models.join(', ')}</p>
            <p><strong>Total Models:</strong> ${data.model_info.total_models}</p>
            <p><strong>Classes:</strong> ${data.model_info.classes}</p>
            <p><strong>Computing Device:</strong> ${data.model_info.device}</p>
        `;
        
        // Update disease info
        const diseaseDetails = document.getElementById('diseaseDetails');
        let treatmentInfo = '';
        
        // Provide disease information based on prediction
        if (diseaseClass.includes('Anthracnose fruit rot')) {
            treatmentInfo = `
                <p><strong>Cause:</strong> Fungal pathogen <em>Colletotrichum</em> species</p>
                <p><strong>Symptoms:</strong> Dark, sunken, circular spots on fruits with orange spore masses.</p>
                <p><strong>Treatment:</strong></p>
                <ul>
                    <li>Remove and destroy infected fruits</li>
                    <li>Apply copper-based fungicides</li>
                    <li>Ensure proper plant spacing for air circulation</li>
                    <li>Avoid overhead watering</li>
                </ul>
                <p><strong>Prevention:</strong> Crop rotation, proper sanitation, resistant varieties</p>
            `;
        } else if (diseaseClass.includes('Anthracnose leaf spot')) {
            treatmentInfo = `
                <p><strong>Cause:</strong> Fungal pathogen <em>Colletotrichum</em> species</p>
                <p><strong>Symptoms:</strong> Small, water-soaked spots on leaves that enlarge and turn brown with yellow halos.</p>
                <p><strong>Treatment:</strong></p>
                <ul>
                    <li>Remove infected leaves</li>
                    <li>Apply fungicides with chlorothalonil or azoxystrobin</li>
                    <li>Improve air circulation</li>
                </ul>
                <p><strong>Prevention:</strong> Mulching, crop rotation, drip irrigation</p>
            `;
        } else if (diseaseClass.includes('Blossom end rot')) {
            treatmentInfo = `
                <p><strong>Cause:</strong> Calcium deficiency, usually due to inconsistent watering</p>
                <p><strong>Symptoms:</strong> Dark, leathery patches on the bottom (blossom end) of fruits.</p>
                <p><strong>Treatment:</strong></p>
                <ul>
                    <li>Maintain consistent soil moisture</li>
                    <li>Apply calcium supplements to soil</li>
                    <li>Mulch around plants</li>
                    <li>Avoid excessive nitrogen fertilization</li>
                </ul>
                <p><strong>Prevention:</strong> Regular watering, proper calcium levels in soil</p>
            `;
        } else if (diseaseClass.includes('Yellow mosaic virus')) {
            treatmentInfo = `
                <p><strong>Cause:</strong> Begomovirus transmitted by whiteflies</p>
                <p><strong>Symptoms:</strong> Yellow mottling or mosaic pattern on leaves, stunted growth.</p>
                <p><strong>Treatment:</strong></p>
                <ul>
                    <li>No cure - remove and destroy infected plants</li>
                    <li>Control whitefly populations with insecticides</li>
                    <li>Use reflective mulches to repel whiteflies</li>
                </ul>
                <p><strong>Prevention:</strong> Resistant varieties, insect pest management, weed control</p>
            `;
        } else if (diseaseClass.includes('Insect damaged leaf')) {
            treatmentInfo = `
                <p><strong>Cause:</strong> Various insect pests like beetles, caterpillars, or aphids</p>
                <p><strong>Symptoms:</strong> Holes, tears, or chewed edges on leaves.</p>
                <p><strong>Treatment:</strong></p>
                <ul>
                    <li>Identify specific pest for targeted treatment</li>
                    <li>Apply appropriate insecticides or organic solutions like neem oil</li>
                    <li>Introduce beneficial insects</li>
                    <li>Remove severely damaged leaves</li>
                </ul>
                <p><strong>Prevention:</strong> Regular monitoring, crop rotation, companion planting</p>
            `;
        } else if (diseaseClass.includes('Fresh')) {
            treatmentInfo = `
                <p><strong>Status:</strong> Healthy plant tissue</p>
                <p><strong>Maintenance:</strong></p>
                <ul>
                    <li>Continue regular watering and fertilization</li>
                    <li>Monitor for early signs of pests or diseases</li>
                    <li>Maintain good air circulation around plants</li>
                    <li>Follow proper pruning practices</li>
                </ul>
                <p><strong>Prevention:</strong> Balanced nutrition, proper spacing, clean tools</p>
            `;
        }
        
        diseaseDetails.innerHTML = treatmentInfo;
        
        // Show all hidden elements
        document.getElementById('allPredictions').style.display = 'block';
        document.getElementById('modelInfo').style.display = 'block';
        document.getElementById('diseaseInfo').style.display = 'block';
        result.style.display = 'block';
    }

    // Handle drag and drop functionality for upload area
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            imageInput.files = e.dataTransfer.files;
            const event = new Event('change');
            imageInput.dispatchEvent(event);
        }
    });

    // For demonstration purposes - allows the app to work without backend
    // Remove this for production with actual backend
    window.demoMode = true;
    if (window.demoMode) {
        submitPrediction = function() {
            // All possible diseases
            const allDiseases = [
                "Anthracnose fruit rot",
                "Anthracnose leaf spot",
                "Blossom end rot",
                "Fresh fruit",
                "Fresh leaf",
                "Insect damaged leaf",
                "Yellow mosaic virus"
            ];
            
            // Get image data to make "prediction" based on image characteristics
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 100;
            canvas.height = 100;
            context.drawImage(previewImg, 0, 0, 100, 100);
            
            // Get image data and calculate some basic characteristics
            let imageData;
            try {
                imageData = context.getImageData(0, 0, 100, 100).data;
            } catch(e) {
                // CORS issue, fallback to random
                console.log("Could not access image data, using random prediction");
                imageData = null;
            }
            
            // Use image characteristics to pick a disease, or random if we can't access the image data
            let diseaseIndex = 0;
            if (imageData) {
                // Use image characteristics to determine "prediction"
                let redSum = 0, greenSum = 0, yellowSum = 0;
                
                // Sample a few pixels
                for (let i = 0; i < imageData.length; i += 40) {
                    const r = imageData[i];
                    const g = imageData[i + 1];
                    const b = imageData[i + 2];
                    
                    redSum += r;
                    greenSum += g;
                    
                    // Detect yellow (high red and green, low blue)
                    if (r > 150 && g > 150 && b < 100) {
                        yellowSum += 1;
                    }
                }
                
                // Logic to determine disease based on image characteristics
                if (yellowSum > 20) {
                    // Lots of yellow, likely Yellow mosaic virus
                    diseaseIndex = 6; // Yellow mosaic virus
                } else if (redSum > greenSum * 1.5) {
                    // Reddish image, likely Anthracnose fruit rot
                    diseaseIndex = 0; // Anthracnose fruit rot
                } else if (greenSum > redSum * 1.5) {
                    // Very green image, likely Fresh leaf
                    diseaseIndex = 4; // Fresh leaf
                } else {
                    // Use timestamp to create some variability
                    diseaseIndex = Math.floor(Date.now() % allDiseases.length);
                }
            } else {
                // Random if we couldn't access image data
                diseaseIndex = Math.floor(Math.random() * allDiseases.length);
            }
            
            // Create confidence value (60-95%)
            const mainConfidence = (60 + Math.floor(Math.random() * 36)).toFixed(2);
            
            // Create random mock results with appropriate confidence values
            const prediction = allDiseases[diseaseIndex];
            const mockAllPredictions = {};
            
            // Distribute the remaining confidence (100 - mainConfidence)
            const remainingConfidence = 100 - parseFloat(mainConfidence);
            let runningTotal = 0;
            
            // Create mock confidence values for other diseases
            allDiseases.forEach((disease, i) => {
                if (i === diseaseIndex) {
                    // This is our main prediction
                    mockAllPredictions[disease] = mainConfidence + "%";
                } else {
                    // Random confidence for others, but proportionally lower
                    const randomConfidence = (remainingConfidence * Math.random()).toFixed(2);
                    runningTotal += parseFloat(randomConfidence);
                    mockAllPredictions[disease] = randomConfidence + "%";
                }
            });
            
            // Create final mock results object
            const mockResults = {
                prediction: prediction,
                confidence: mainConfidence + "%",
                all_predictions: mockAllPredictions,
                model_info: {
                    architecture: "PyTorch Ensemble (Demo Mode)",
                    base_models: ["ResNet50", "EfficientNetB3", "DenseNet121", "InceptionV3", "ConvNeXt", "Swin Tiny"],
                    total_models: 6,
                    device: "Static Deployment (Demo Mode)",
                    classes: 7
                }
            };
            
            setTimeout(() => {
                displayResults(mockResults);
            }, 500);
            
            setTimeout(() => {
                displayResults(mockResults);
            }, 500);
        };
    }
});
