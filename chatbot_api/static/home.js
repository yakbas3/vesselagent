document.addEventListener('DOMContentLoaded', async () => {
    const container = document.getElementById('averagesTableContainer');

    try {
        const response = await fetch('/home_data');
        if (!response.ok) {
            let errorText = `Error fetching data: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorText = `Error: ${errorData.detail || errorText}`;
            } catch (e) { /* Ignore if body isn't JSON */ }
            throw new Error(errorText);
        }

        const averages = await response.json();

        // Clear loading message
        container.innerHTML = ''; 

        // Create and populate table
        const table = document.createElement('table');
        const thead = table.createTHead();
        const tbody = table.createTBody();
        const headerRow = thead.insertRow();
        const header1 = document.createElement('th');
        const header2 = document.createElement('th');
        header1.textContent = 'Metric';
        header2.textContent = 'Average Value';
        headerRow.appendChild(header1);
        headerRow.appendChild(header2);

        for (const [metric, avgValue] of Object.entries(averages)) {
            const row = tbody.insertRow();
            const cell1 = row.insertCell();
            const cell2 = row.insertCell();
            cell1.textContent = metric;
            // Format numbers nicely, handle nulls
            cell2.textContent = (avgValue === null || avgValue === undefined) 
                                ? 'N/A' 
                                : typeof avgValue === 'number' ? avgValue.toFixed(2) : avgValue;
        }

        container.appendChild(table);

    } catch (error) {
        console.error("Failed to load averages:", error);
        container.innerHTML = `<p style="color: red;">Failed to load averages: ${error.message}</p>`;
    }
}); 