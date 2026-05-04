document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultContainer = document.getElementById('resultContainer');
    
    // Set default date to today
    document.getElementById('PredictionDate').valueAsDate = new Date();

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Feedback
        submitBtn.disabled = true;
        submitBtn.textContent = 'CALCULATING...';
        resultContainer.classList.add('hidden');

        // Collect Form Data
        const formData = new FormData(form);
        const payload = {};
        
        formData.forEach((value, key) => {
            if (key === 'isfloodedArea' || key === 'IsTsunmaiAlertedArea') {
                payload[key] = true;
            } else {
                payload[key] = value;
            }
        });
        
        // Handle unchecked boxes
        if (!payload['isfloodedArea']) payload['isfloodedArea'] = false;
        if (!payload['IsTsunmaiAlertedArea']) payload['IsTsunmaiAlertedArea'] = false;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                // Update Results
                document.getElementById('predictedPrice').textContent = data.prediction.toLocaleString();
                document.getElementById('priceRange').textContent = 
                    `Rs. ${data.lower_bound.toLocaleString()} - Rs. ${data.upper_bound.toLocaleString()}`;
                document.getElementById('resAccuracy').textContent = `${data.accuracy.toFixed(2)}%`;
                document.getElementById('resModel').textContent = data.model_name;

                // Show Container
                resultContainer.classList.remove('hidden');
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An unexpected error occurred. Please check console.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'CALCULATE ESTIMATED VALUE';
        }
    });
});
