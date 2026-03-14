import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Calendar, Cloud, Sun, Droplets, Thermometer } from 'lucide-react';

// Generate sample meteorological data for 2024
const generateMeteoData = () => {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return months.map((month, i) => ({
    month,
    ghi: parseFloat((4.5 + Math.sin(i / 12 * Math.PI * 2) * 1.5 + Math.random() * 0.5).toFixed(2)),
    temperature: parseFloat((26 + Math.sin(i / 12 * Math.PI * 2) * 2 + Math.random() * 0.5).toFixed(1)),
    humidity: parseFloat((70 + Math.sin(i / 12 * Math.PI * 2 + Math.PI) * 10 + Math.random() * 5).toFixed(0)),
    cloudCover: parseFloat((4 + Math.sin(i / 12 * Math.PI * 2 + Math.PI) * 1.5 + Math.random() * 0.5).toFixed(1)),
    sunshineHours: parseFloat((6 + Math.sin(i / 12 * Math.PI * 2) * 2 + Math.random() * 0.5).toFixed(1)),
  }));
};

const meteoData = generateMeteoData();

// Generate daily data for selected month
const generateDailyData = (month: number) => {
  const days = 30;
  return Array.from({ length: days }, (_, i) => ({
    day: i + 1,
    ghi: parseFloat((4 + Math.random() * 3).toFixed(2)),
    clearSkyRatio: parseFloat((0.6 + Math.random() * 0.35).toFixed(3)),
    temperature: parseFloat((25 + Math.random() * 6).toFixed(1)),
  }));
};

export function DataVisualization() {
  const [selectedMonth, setSelectedMonth] = useState(0);
  const [activeMetric, setActiveMetric] = useState<'ghi' | 'temperature' | 'humidity' | 'cloudCover'>('ghi');
  
  const dailyData = generateDailyData(selectedMonth);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h2 className="text-xl mb-2 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-blue-500" />
          Meteorological Data Insights
        </h2>
        <p className="text-sm text-gray-600">
          NASA POWER satellite data for Davao City (7.191°N, 125.455°E) • 2024
        </p>
      </div>

      {/* Annual Overview */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg">Annual Meteorological Trends</h3>
          <select
            value={activeMetric}
            onChange={(e) => setActiveMetric(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="ghi">Global Horizontal Irradiance</option>
            <option value="temperature">Temperature</option>
            <option value="humidity">Humidity</option>
            <option value="cloudCover">Cloud Cover</option>
          </select>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={meteoData}>
            <defs>
              <linearGradient id="colorMetric" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="month" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Area
              type="monotone"
              dataKey={activeMetric}
              stroke="#3b82f6"
              fillOpacity={1}
              fill="url(#colorMetric)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Monthly Statistics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-orange-50 to-yellow-50 border border-orange-200 rounded-xl p-5 shadow-md">
          <Sun className="w-8 h-8 text-orange-500 mb-3" />
          <p className="text-sm text-orange-800 mb-1">Avg GHI</p>
          <p className="text-2xl text-orange-900 mb-1">5.24</p>
          <p className="text-xs text-orange-700">kWh/m²/day</p>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-orange-50 border border-red-200 rounded-xl p-5 shadow-md">
          <Thermometer className="w-8 h-8 text-red-500 mb-3" />
          <p className="text-sm text-red-800 mb-1">Avg Temperature</p>
          <p className="text-2xl text-red-900 mb-1">27.3°C</p>
          <p className="text-xs text-red-700">T2M</p>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-5 shadow-md">
          <Droplets className="w-8 h-8 text-blue-500 mb-3" />
          <p className="text-sm text-blue-800 mb-1">Avg Humidity</p>
          <p className="text-2xl text-blue-900 mb-1">76.2%</p>
          <p className="text-xs text-blue-700">RH2M</p>
        </div>

        <div className="bg-gradient-to-br from-gray-50 to-slate-50 border border-gray-200 rounded-xl p-5 shadow-md">
          <Cloud className="w-8 h-8 text-gray-500 mb-3" />
          <p className="text-sm text-gray-800 mb-1">Avg Cloud Cover</p>
          <p className="text-2xl text-gray-900 mb-1">4.2</p>
          <p className="text-xs text-gray-700">oktas</p>
        </div>
      </div>

      {/* Daily Data Viewer */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg flex items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-500" />
            Daily Data Explorer
          </h3>
          <select
            value={selectedMonth}
            onChange={(e) => setSelectedMonth(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {meteoData.map((m, i) => (
              <option key={i} value={i}>{m.month} 2024</option>
            ))}
          </select>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={dailyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="day" 
              tick={{ fontSize: 12 }}
              label={{ value: 'Day of Month', position: 'insideBottom', offset: -5, fontSize: 12 }}
            />
            <YAxis 
              yAxisId="left"
              tick={{ fontSize: 12 }}
              label={{ value: 'GHI (kWh/m²/day)', angle: -90, position: 'insideLeft', fontSize: 12 }}
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 12 }}
              label={{ value: 'Temperature (°C)', angle: 90, position: 'insideRight', fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line yAxisId="left" type="monotone" dataKey="ghi" stroke="#f97316" strokeWidth={2} dot={false} name="GHI" />
            <Line yAxisId="right" type="monotone" dataKey="temperature" stroke="#3b82f6" strokeWidth={2} dot={false} name="Temperature" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Feature Engineering Summary */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4">Engineered Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="text-sm mb-3">Temporal Features</h4>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>Month (sin/cos):</strong> Cyclical encoding for seasonality</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span><strong>Season Indicator:</strong> Dry (Dec-May) vs Rainy (Jun-Nov)</span>
              </li>
            </ul>
          </div>

          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="text-sm mb-3">Meteorological Features</h4>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>Sunshine Hours:</strong> Hours with irradiance &gt; 120 W/m²</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-orange-500">•</span>
                <span><strong>Clear Sky Ratio:</strong> GHI_actual / GHI_clear_sky</span>
              </li>
            </ul>
          </div>

          <div className="border border-gray-200 rounded-lg p-4 md:col-span-2">
            <h4 className="text-sm mb-3">Topographical Features (Solar Exposure Index)</h4>
            <p className="text-sm text-gray-700 mb-2">
              <strong>SEI = </strong>(Orientation Score) × (Area) × (1 - Shading Factor) × (Tilt Factor)
            </p>
            <ul className="grid sm:grid-cols-2 gap-2 text-sm text-gray-700">
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span><strong>Orientation:</strong> cos(azimuth - 180°)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span><strong>Area:</strong> Rooftop area from OSM (m²)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span><strong>Shading:</strong> Based on building density</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span><strong>Tilt:</strong> cos(|roof_tilt - 7.2°|)</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrendingUp({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  );
}
