document.addEventListener('DOMContentLoaded', async () => {
    const container = document.getElementById('insightsContainer');

    function displayResults(data) {
        container.innerHTML = ''; // Clear loading message

        if (!data || !data.regression_insights || data.regression_insights.length === 0) {
            container.innerHTML = '<p>No regression insights were generated.</p>';
            return;
        }

        data.regression_insights.forEach(insight => {
            const block = document.createElement('div');
            block.classList.add('analysis-block');

            const title = document.createElement('h3');
            title.textContent = insight.analysis_name || 'Unnamed Analysis';
            block.appendChild(title);

            const statusPara = document.createElement('p');
            statusPara.innerHTML = `Status: <strong class="status-${(insight.status || 'failed').toLowerCase()}">${insight.status}</strong>`;
            block.appendChild(statusPara);

            if (insight.details) {
                // Display details nicely, maybe format interpretation separately
                if (typeof insight.details === 'string') { 
                    const detailsPara = document.createElement('p');
                    detailsPara.textContent = `Details: ${insight.details}`;
                    block.appendChild(detailsPara);
                } else if (typeof insight.details === 'object') {
                    const interpretationPara = document.createElement('p');
                    interpretationPara.innerHTML = `<strong>Interpretation:</strong> ${insight.details.interpretation || 'N/A'}`;
                    block.appendChild(interpretationPara);
                    
                    const detailsPre = document.createElement('pre');
                    // Create a copy without interpretation for cleaner JSON display
                    const detailsToDisplay = {...insight.details};
                    delete detailsToDisplay.interpretation; 
                    detailsPre.textContent = JSON.stringify(detailsToDisplay, null, 2);
                    block.appendChild(detailsPre);
                }
            } else {
                const noDetails = document.createElement('p');
                noDetails.textContent = 'No further details available.';
                block.appendChild(noDetails);
            }
            
            container.appendChild(block);
        });
    }

    // --- Fetch the data --- 
    try {
        const response = await fetch('/insights'); // Fetch from the new endpoint
        if (!response.ok) {
            let errorText = `Error fetching insights: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorText = `Error: ${errorData.detail || errorText}`;
            } catch (e) { /* Ignore */ }
            throw new Error(errorText);
        }
        const insightsData = await response.json();
        displayResults(insightsData);

    } catch (error) {
        console.error("Failed to load insights:", error);
        container.innerHTML = `<p style="color: red;">Failed to load insights: ${error.message}</p>`;
    }
}); 