import { Sun, Cloud, Droplets, Thermometer, TrendingUp, Award } from 'lucide-react';
import { MetricCard } from './MetricCard';
import { PerformanceChart } from './PerformanceChart';
import { FeatureImportanceChart } from './FeatureImportanceChart';

export function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-orange-500 rounded-2xl p-6 text-white shadow-xl">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl mb-2">Solar Energy Forecasting</h2>
            <p className="text-blue-100 text-sm max-w-2xl">
              Feature-Importance-Aware AdaBoost Regression for rooftop solar potential in Davao City
            </p>
          </div>
          <Sun className="w-12 h-12 text-yellow-300 animate-pulse" />
        </div>
        
        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <Sun className="w-4 h-4 text-yellow-300" />
              <span className="text-xs text-blue-100">Avg GHI</span>
            </div>
            <p className="text-xl">5.24</p>
            <p className="text-xs text-blue-100">kWh/m²/day</p>
          </div>
          
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <Building2Icon className="w-4 h-4 text-yellow-300" />
              <span className="text-xs text-blue-100">Rooftops</span>
            </div>
            <p className="text-xl">10,000</p>
            <p className="text-xs text-blue-100">analyzed</p>
          </div>
          
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <Thermometer className="w-4 h-4 text-yellow-300" />
              <span className="text-xs text-blue-100">Avg Temp</span>
            </div>
            <p className="text-xl">27.3°C</p>
            <p className="text-xs text-blue-100">2024</p>
          </div>
          
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <Cloud className="w-4 h-4 text-yellow-300" />
              <span className="text-xs text-blue-100">Cloud Cover</span>
            </div>
            <p className="text-xl">4.2</p>
            <p className="text-xs text-blue-100">oktas avg</p>
          </div>
        </div>
      </div>

      {/* Model Performance Metrics */}
      <div>
        <h3 className="text-lg mb-3 flex items-center gap-2">
          <Award className="w-5 h-5 text-orange-500" />
          FI-AdaBoost Performance
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetricCard
            title="R² Score"
            value="0.9867"
            subtitle="Coefficient of Determination"
            trend="+2.1%"
            trendUp={true}
            color="blue"
          />
          <MetricCard
            title="RMSE"
            value="0.182"
            subtitle="kWh/m²/day"
            trend="-18.5%"
            trendUp={true}
            color="green"
          />
          <MetricCard
            title="MAE"
            value="0.134"
            subtitle="kWh/m²/day"
            trend="-15.2%"
            trendUp={true}
            color="orange"
          />
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PerformanceChart />
        <FeatureImportanceChart />
      </div>

      {/* Meteorological Data */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <Cloud className="w-5 h-5 text-blue-500" />
          Current Meteorological Conditions
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <Sun className="w-8 h-8 text-orange-500 mx-auto mb-2" />
            <p className="text-sm text-gray-600">GHI</p>
            <p className="text-xl">5.83</p>
            <p className="text-xs text-gray-500">kWh/m²/day</p>
          </div>
          
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <Thermometer className="w-8 h-8 text-red-500 mx-auto mb-2" />
            <p className="text-sm text-gray-600">Temperature</p>
            <p className="text-xl">28.5°C</p>
            <p className="text-xs text-gray-500">T2M</p>
          </div>
          
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <Droplets className="w-8 h-8 text-blue-500 mx-auto mb-2" />
            <p className="text-sm text-gray-600">Humidity</p>
            <p className="text-xl">76%</p>
            <p className="text-xs text-gray-500">RH2M</p>
          </div>
          
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Cloud className="w-8 h-8 text-gray-500 mx-auto mb-2" />
            <p className="text-sm text-gray-600">Cloud Cover</p>
            <p className="text-xl">3.8</p>
            <p className="text-xs text-gray-500">oktas</p>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200">
          <h4 className="text-sm text-green-800 mb-2">Model Improvement</h4>
          <p className="text-3xl text-green-700 mb-1">18.5%</p>
          <p className="text-sm text-green-600">RMSE reduction vs baseline AdaBoost</p>
        </div>
        
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
          <h4 className="text-sm text-blue-800 mb-2">Solar Potential</h4>
          <p className="text-3xl text-blue-700 mb-1">91,000 MW</p>
          <p className="text-sm text-blue-600">Estimated rooftop capacity (Philippines)</p>
        </div>
      </div>
    </div>
  );
}

function Building2Icon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
    </svg>
  );
}
