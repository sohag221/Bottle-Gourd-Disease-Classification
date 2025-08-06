/* For GitHub Pages deployment - making sure styles are properly applied */

// Add this to verify that the site is loading CSS correctly
document.addEventListener('DOMContentLoaded', function() {
    console.log("Page loaded successfully!");
    
    // Check if CSS is loaded
    const isCssLoaded = Array.from(document.styleSheets).some(sheet => {
        try {
            return sheet.href && sheet.href.includes('style.css');
        } catch (e) {
            return false;
        }
    });
    
    console.log("CSS loaded:", isCssLoaded);
    
    // If CSS is not loaded, we'll add an alert for debugging
    if (!isCssLoaded) {
        const alert = document.createElement('div');
        alert.style.padding = '10px';
        alert.style.backgroundColor = 'red';
        alert.style.color = 'white';
        alert.style.position = 'fixed';
        alert.style.top = '0';
        alert.style.left = '0';
        alert.style.right = '0';
        alert.style.zIndex = '9999';
        alert.innerHTML = 'Warning: CSS not loaded properly! Check browser console.';
        document.body.prepend(alert);
    }

    // Setup demo mode by default
    window.demoMode = true;
    
    // Add demo mode banner
    const demoBanner = document.createElement('div');
    demoBanner.style.padding = '6px 10px';
    demoBanner.style.backgroundColor = '#007bff';
    demoBanner.style.color = 'white';
    demoBanner.style.position = 'fixed';
    demoBanner.style.bottom = '0';
    demoBanner.style.left = '0';
    demoBanner.style.right = '0';
    demoBanner.style.zIndex = '9999';
    demoBanner.style.textAlign = 'center';
    demoBanner.style.fontSize = '14px';
    demoBanner.innerHTML = '⚠️ Demo Mode: Predictions are simulated based on image analysis. For actual AI predictions, run the app locally with Python backend.';
    
    // Add the banner after page load
    setTimeout(() => {
        document.body.appendChild(demoBanner);
    }, 1000);
});
