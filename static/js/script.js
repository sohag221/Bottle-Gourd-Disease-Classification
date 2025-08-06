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
            // Create demo data
            const mockResults = {
                prediction: "Yellow mosaic virus",
                confidence: "93.45%",
                all_predictions: {
                    "Yellow mosaic virus": "93.45%",
                    "Anthracnose leaf spot": "4.32%",
                    "Insect damaged leaf": "1.05%",
                    "Anthracnose fruit rot": "0.58%",
                    "Blossom end rot": "0.34%",
                    "Fresh leaf": "0.21%",
                    "Fresh fruit": "0.05%"
                },
                model_info: {
                    architecture: "PyTorch Ensemble",
                    base_models: ["ResNet50", "EfficientNetB3", "DenseNet121", "InceptionV3", "ConvNeXt", "Swin Tiny"],
                    total_models: 6,
                    device: "CPU",
                    classes: 7
                }
            };
            
            setTimeout(() => {
                displayResults(mockResults);
            }, 500);
        };
    }
});
