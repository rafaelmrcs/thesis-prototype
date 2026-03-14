import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const data = [
  { name: 'Clear Sky Ratio', importance: 0.37 },
  { name: 'Sunshine Hours', importance: 0.37 },
  { name: 'Solar Exposure Index', importance: 0.26 },
  { name: 'GHI', importance: 0.21 },
  { name: 'Cloud Cover', importance: 0.19 },
  { name: 'Temperature', importance: 0.15 },
  { name: 'Humidity', importance: 0.12 },
  { name: 'Rooftop Area', importance: 0.10 },
];

const COLORS = ['#f97316', '#fb923c', '#fdba74', '#fed7aa', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'];

export function FeatureImportanceChart() {
  return (
    <div className="bg-white rounded-xl p-6 shadow-md">
      <h3 className="text-lg mb-4">Feature Importance (FI-AdaBoost)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, 0.4]} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={120} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-500 mt-3 text-center">
        Normalized feature importance scores from decision tree weak learners
      </p>
    </div>
  );
}
