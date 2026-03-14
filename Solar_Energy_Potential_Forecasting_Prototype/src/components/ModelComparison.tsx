import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { CheckCircle2, AlertCircle, TrendingUp } from 'lucide-react';

// Generate sample prediction data
const generatePredictionData = () => {
  const data = [];
  for (let i = 0; i < 30; i++) {
    const actual = 4 + Math.random() * 3;
    data.push({
      day: i + 1,
      actual: parseFloat(actual.toFixed(2)),
      adaboost: parseFloat((actual + (Math.random() - 0.5) * 0.5).toFixed(2)),
      fiAdaboost: parseFloat((actual + (Math.random() - 0.5) * 0.35).toFixed(2)),
    });
  }
  return data;
};

const predictionData = generatePredictionData();

// Generate scatter data for actual vs predicted
const scatterData = predictionData.map(d => ({
  actual: d.actual,
  adaboost: d.adaboost,
  fiAdaboost: d.fiAdaboost,
}));

export function ModelComparison() {
  const [activeTab, setActiveTab] = useState<'metrics' | 'timeseries' | 'scatter'>('metrics');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h2 className="text-xl mb-2">Model Performance Comparison</h2>
        <p className="text-sm text-gray-600">
          Baseline AdaBoost Regression vs Feature-Importance-Aware AdaBoost (FI-AdaBoost)
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
            activeTab === 'metrics'
              ? 'bg-gradient-to-r from-blue-500 to-orange-500 text-white shadow-md'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Performance Metrics
        </button>
        <button
          onClick={() => setActiveTab('timeseries')}
          className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
            activeTab === 'timeseries'
              ? 'bg-gradient-to-r from-blue-500 to-orange-500 text-white shadow-md'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Time Series
        </button>
        <button
          onClick={() => setActiveTab('scatter')}
          className={`px-4 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
            activeTab === 'scatter'
              ? 'bg-gradient-to-r from-blue-500 to-orange-500 text-white shadow-md'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Actual vs Predicted
        </button>
      </div>

      {/* Content */}
      {activeTab === 'metrics' && (
        <div className="space-y-6">
          {/* Comparison Table */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm">Metric</th>
                    <th className="px-4 py-3 text-center text-sm">AdaBoost</th>
                    <th className="px-4 py-3 text-center text-sm">FI-AdaBoost</th>
                    <th className="px-4 py-3 text-center text-sm">Improvement</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  <tr className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm">R² Score</td>
                    <td className="px-4 py-3 text-center text-sm">0.9659</td>
                    <td className="px-4 py-3 text-center text-sm">0.9867</td>
                    <td className="px-4 py-3 text-center">
                      <span className="inline-flex items-center gap-1 text-sm text-green-600">
                        <TrendingUp className="w-4 h-4" />
                        +2.1%
                      </span>
                    </td>
                  </tr>
                  <tr className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm">RMSE (kWh/m²/day)</td>
                    <td className="px-4 py-3 text-center text-sm">0.223</td>
                    <td className="px-4 py-3 text-center text-sm">0.182</td>
                    <td className="px-4 py-3 text-center">
                      <span className="inline-flex items-center gap-1 text-sm text-green-600">
                        <TrendingUp className="w-4 h-4" />
                        -18.5%
                      </span>
                    </td>
                  </tr>
                  <tr className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm">MAE (kWh/m²/day)</td>
                    <td className="px-4 py-3 text-center text-sm">0.158</td>
                    <td className="px-4 py-3 text-center text-sm">0.134</td>
                    <td className="px-4 py-3 text-center">
                      <span className="inline-flex items-center gap-1 text-sm text-green-600">
                        <TrendingUp className="w-4 h-4" />
                        -15.2%
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Key Findings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-green-50 border border-green-200 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-sm text-green-900 mb-2">FI-AdaBoost Advantages</h3>
                  <ul className="text-sm text-green-800 space-y-1">
                    <li>• Feature-aware weight updates</li>
                    <li>• Better handling of outliers</li>
                    <li>• Improved interpretability</li>
                    <li>• Lower prediction errors</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-sm text-blue-900 mb-2">Implementation Notes</h3>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Uses decision tree weak learners</li>
                    <li>• Max depth: 3 (prevents overfitting)</li>
                    <li>• Bayesian optimization for tuning</li>
                    <li>• TimeSeriesSplit validation</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'timeseries' && (
        <div className="bg-white rounded-xl p-6 shadow-md">
          <h3 className="text-lg mb-4">30-Day Forecast Comparison</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={predictionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="day" tick={{ fontSize: 12 }} label={{ value: 'Day', position: 'insideBottom', offset: -5, fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} label={{ value: 'GHI (kWh/m²/day)', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={2} dot={false} name="Actual" />
              <Line type="monotone" dataKey="adaboost" stroke="#3b82f6" strokeWidth={2} dot={false} name="AdaBoost" strokeDasharray="5 5" />
              <Line type="monotone" dataKey="fiAdaboost" stroke="#f97316" strokeWidth={2} dot={false} name="FI-AdaBoost" />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-3 text-center">
            FI-AdaBoost predictions align more closely with actual values
          </p>
        </div>
      )}

      {activeTab === 'scatter' && (
        <div className="bg-white rounded-xl p-6 shadow-md">
          <h3 className="text-lg mb-4">Actual vs Predicted Values</h3>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                type="number" 
                dataKey="actual" 
                name="Actual" 
                tick={{ fontSize: 12 }}
                label={{ value: 'Actual GHI (kWh/m²/day)', position: 'insideBottom', offset: -5, fontSize: 12 }}
              />
              <YAxis 
                type="number" 
                dataKey="fiAdaboost" 
                name="Predicted" 
                tick={{ fontSize: 12 }}
                label={{ value: 'Predicted GHI (kWh/m²/day)', angle: -90, position: 'insideLeft', fontSize: 12 }}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Scatter name="AdaBoost" data={scatterData} fill="#3b82f6" dataKey="adaboost" />
              <Scatter name="FI-AdaBoost" data={scatterData} fill="#f97316" dataKey="fiAdaboost" />
              <Line type="linear" dataKey="actual" stroke="#10b981" strokeWidth={2} dot={false} />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-3 text-center">
            Points closer to the diagonal line indicate better predictions
          </p>
        </div>
      )}
    </div>
  );
}
