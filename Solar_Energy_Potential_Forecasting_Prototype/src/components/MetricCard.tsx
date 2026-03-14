import { TrendingUp, TrendingDown } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  subtitle: string;
  trend?: string;
  trendUp?: boolean;
  color?: 'blue' | 'green' | 'orange' | 'purple';
}

export function MetricCard({ title, value, subtitle, trend, trendUp, color = 'blue' }: MetricCardProps) {
  const colorClasses = {
    blue: 'from-blue-50 to-blue-100 border-blue-200 text-blue-700',
    green: 'from-green-50 to-green-100 border-green-200 text-green-700',
    orange: 'from-orange-50 to-orange-100 border-orange-200 text-orange-700',
    purple: 'from-purple-50 to-purple-100 border-purple-200 text-purple-700',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-xl p-5 border shadow-md`}>
      <h4 className="text-sm opacity-80 mb-2">{title}</h4>
      <p className="text-3xl mb-1">{value}</p>
      <p className="text-xs opacity-70">{subtitle}</p>
      
      {trend && (
        <div className={`flex items-center gap-1 mt-3 text-sm ${trendUp ? 'text-green-600' : 'text-red-600'}`}>
          {trendUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          <span>{trend} vs baseline</span>
        </div>
      )}
    </div>
  );
}
