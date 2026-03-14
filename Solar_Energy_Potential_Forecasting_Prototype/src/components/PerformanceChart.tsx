import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  {
    name: 'R² Score',
    'AdaBoost': 0.9659,
    'FI-AdaBoost': 0.9867,
  },
  {
    name: 'RMSE',
    'AdaBoost': 0.223,
    'FI-AdaBoost': 0.182,
  },
  {
    name: 'MAE',
    'AdaBoost': 0.158,
    'FI-AdaBoost': 0.134,
  },
];

export function PerformanceChart() {
  return (
    <div className="bg-white rounded-xl p-6 shadow-md">
      <h3 className="text-lg mb-4">Model Performance Comparison</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '12px',
            }}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          <Bar dataKey="AdaBoost" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          <Bar dataKey="FI-AdaBoost" fill="#f97316" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-500 mt-3 text-center">
        Lower is better for RMSE and MAE • Higher is better for R²
      </p>
    </div>
  );
}
