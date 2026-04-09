// Initialize Lucide icons
lucide.createIcons();

let chartInstance = null;
let globalData = {};

async function loadData() {
    try {
        const response = await fetch('../backend/data/liveData.json');
        const data = await response.json();
        globalData = data.signals;

        document.getElementById('generated-at').innerText = `Generated: ${new Date(data.generated_at).toLocaleString()}`;

        populateSelect();
        const tickers = Object.keys(globalData);
        if (tickers.length > 0) {
            updateDashboard(tickers[0]);
        }
    } catch (error) {
        console.error("Error loading JSON:", error);
        document.getElementById('live-status').innerText = "ERROR LOADING DATA";
        document.getElementById('live-status').classList.replace('text-emerald-500', 'text-red-500');
    }
}

function populateSelect() {
    const select = document.getElementById('stock-select');
    Object.keys(globalData).forEach(ticker => {
        const option = document.createElement('option');
        option.value = ticker;
        option.textContent = ticker;
        select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
        updateDashboard(e.target.value);
    });
}

function updateDashboard(ticker) {
    const data = globalData[ticker];
    if (!data) return;

    // Update Card Classes
    const card = document.getElementById('signal-card');
    card.className = `bg-[#0a0a0a] border border-gray-800 rounded-xl p-6 relative overflow-hidden tier-${data.tier}`;

    // Text values
    document.getElementById('signal-direction').innerText = data.signal;
    document.getElementById('signal-direction').className = `text-3xl font-bold tracking-tight flex items-center gap-2 ${data.signal === 'BUY' ? 'text-emerald-500' : data.signal === 'HOLD' ? 'text-gray-400' : 'text-red-500'}`;

    document.getElementById('signal-tier').innerText = data.tier;
    document.getElementById('signal-strategy').innerText = data.strategy || 'No active strategy';
    document.getElementById('signal-wr').innerText = `${(data.historical_win_rate * 100).toFixed(1)}%`;
    document.getElementById('signal-hold').innerText = `T+${data.hold_days}`;

    // Prices
    document.getElementById('price-current').innerText = data.current_price.toFixed(2);
    document.getElementById('price-entry').innerText = isNaN(data.entry_zone[0]) ? 'N/A' : `${data.entry_zone[0].toFixed(2)} - ${data.entry_zone[1].toFixed(2)}`;
    document.getElementById('price-target').innerText = isNaN(data.upper_bound) ? 'N/A' : data.upper_bound.toFixed(2);
    document.getElementById('price-stop').innerText = isNaN(data.stop_loss) ? 'N/A' : data.stop_loss.toFixed(2);

    // Indicators
    document.getElementById('ind-rsi').innerText = data.rsi_14.toFixed(1);
    document.getElementById('ind-adx').innerText = data.adx_14.toFixed(1);
    document.getElementById('ind-vol').innerText = `${data.rel_volume.toFixed(1)}x`;
    document.getElementById('ind-risk').innerText = data.t25_risk;
    document.getElementById('ind-risk').className = `font-mono text-lg ${data.t25_risk === 'HIGH' ? 'text-red-500' : data.t25_risk === 'LOW' ? 'text-emerald-500' : 'text-yellow-500'}`;

    updateChart(data);
}

function updateChart(signalData) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    // Mocking historical data points based on current price for visual effect
    // In a real app, we'd load the historical CSV for the chart
    const labels = ['T-5', 'T-4', 'T-3', 'T-2', 'T-1', 'Current', 'Target'];
    const prices = [
        signalData.current_price * 0.95,
        signalData.current_price * 0.96,
        signalData.current_price * 0.94,
        signalData.current_price * 0.98,
        signalData.current_price * 0.99,
        signalData.current_price,
        isNaN(signalData.upper_bound) ? signalData.current_price : signalData.upper_bound
    ];

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price',
                data: prices,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: '#1f2937' }, ticks: { color: '#9ca3af' } },
                y: { grid: { color: '#1f2937' }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

// Init
loadData();
