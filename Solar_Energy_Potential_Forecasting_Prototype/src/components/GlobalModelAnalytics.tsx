import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import {
 Bar,
 BarChart,
 CartesianGrid,
 Legend,
 LabelList,
 Line,
 LineChart,
 ReferenceLine,
 ResponsiveContainer,
 Scatter,
 ScatterChart,
 Tooltip,
 XAxis,
 YAxis,
} from 'recharts';
import { RefreshCw, TrendingDown, TrendingUp } from 'lucide-react';
import { fetchBackendJson, getBackendBaseUrl } from '../lib/backend';


// ─── Interfaces ──────────────────────────────────────────────────────────────


interface MetricsSummary {
 rmse: number;
 mae: number;
 r2: number;
}


interface MetricsComparison {
 baseline: MetricsSummary;
 fiAdaBoost: MetricsSummary;
 rmseImprovementPct: number;
 maeImprovementPct: number;
 r2ImprovementPct: number;
}


interface PredictedActualPoint {
 actual: number;
 predicted: number;
 date: string;
}


interface TrainingAnalyticsData {
 performanceMetricsComparison: MetricsComparison;
 predictedVsActual: {
   baseline: PredictedActualPoint[];
   fiAdaBoost: PredictedActualPoint[];
   domainMin: number;
   domainMax: number;
 };
 errorDistribution: Array<{
   bucket: string;
   baseline: number;
   fiAdaBoost: number;
 }>;
 residualSummary: {
   baselineStd: number;
   fiStd: number;
   baselineMae: number;
   fiMae: number;
 };
}


interface CVFoldMetric {
 fold: number;
 baseline_rmse: number;
 baseline_mae: number;
 baseline_r2: number;
 fi_rmse: number;
 fi_mae: number;
 fi_r2: number;
}


interface CVMetricsData {
 cv_fold_metrics: CVFoldMetric[];
 average_metrics: {
   baseline: MetricsSummary;
   fiAdaBoost: MetricsSummary;
 };
}


interface SpatialCVFold {
	 fold: number;
	 baseline_train_rmse: number;
	 baseline_val_rmse: number;
	 baseline_val_mae: number;
	 baseline_train_r2: number;
	 baseline_val_r2: number;
	 fi_train_rmse: number;
	 fi_val_rmse: number;
	 fi_val_mae: number;
	 fi_train_r2: number;
	 fi_val_r2: number;
}


interface SpatialCVData {
 folds: SpatialCVFold[];
	 average: {
	   baseline_val_rmse: number;
	   baseline_val_mae: number;
	   baseline_val_r2: number;
	   fi_val_rmse: number;
	   fi_val_mae: number;
	   fi_val_r2: number;
	 };
}


interface DMTestResult {
 dm_statistic: number;
 p_value: number;
 significant: boolean;
 interpretation: string;
 n_samples: number;
 mean_loss_diff_j2: number;
}


interface DMTestData {
 spatial: DMTestResult;
 daily: DMTestResult;
}


interface DailyMetricRow {
 model: string;
 split: string;
 rmse_j: number;
 mae_j: number;
 r2: number;
}


interface DailyMetricsData {
 results: DailyMetricRow[];
}


interface SplitInfoData {
	 total_samples: number;
	 train_samples: number;
	 test_samples: number;
	 test_fraction: number;
	 split_method: string;
	 random_seed: number;
	 target_col: string;
}


interface FeatureWeightRow {
	 rank: number;
	 feature: string;
	 feature_weight: number;
	 feature_weight_percent: number;
	 source: string;
}


interface FeatureWeightsData {
	 weights: FeatureWeightRow[];
	 source_file: string;
	 baseline_weights: FeatureWeightRow[];
	 baseline_source_file: string;
}


const KWH_TO_J = 3_600_000;


function formatExact(value: number, fractionDigits = 6): string {
 return value.toLocaleString('en-US', {
   minimumFractionDigits: fractionDigits,
   maximumFractionDigits: fractionDigits,
   useGrouping: false,
 });
}


function formatResultsErrorMetric(valueInKwh: number): string {
 return formatExact(valueInKwh * KWH_TO_J, 6);
}


function formatRawR2(value: number): string {
 return formatExact(value, 6);
}


function formatPercent(value: number, fractionDigits = 6): string {
 return `${formatExact(value, fractionDigits)}%`;
}


function formatGrouped(value: number, fractionDigits = 2): string {
 return value.toLocaleString('en-US', {
   minimumFractionDigits: fractionDigits,
   maximumFractionDigits: fractionDigits,
 });
}


function formatR2Percent(value: number, fractionDigits = 4): string {
	 return formatPercent(value * 100, fractionDigits);
}


function formatFeatureName(value: string): string {
	 return value
	   .replace(/_/g, ' ')
	   .replace(/\b\w/g, (char) => char.toUpperCase());
}


function formatSmallNumber(value: number, fractionDigits = 6): string {
	 if (value !== 0 && Math.abs(value) < 10 ** -fractionDigits) {
	   return value.toExponential(3);
	 }
	 return formatExact(value, fractionDigits);
}


function formatSmallPercent(value: number, fractionDigits = 4): string {
	 if (value !== 0 && Math.abs(value) < 10 ** -fractionDigits) {
	   return `${value.toExponential(3)}%`;
	 }
	 return formatPercent(value, fractionDigits);
}


// ─── Main component ───────────────────────────────────────────────────────────


export function GlobalModelAnalytics() {
 const [trainingAnalytics, setTrainingAnalytics] = useState<TrainingAnalyticsData | null>(null);
 const [cvMetrics, setCvMetrics] = useState<CVMetricsData | null>(null);
 const [spatialCv, setSpatialCv] = useState<SpatialCVData | null>(null);
	 const [dmTest, setDmTest] = useState<DMTestData | null>(null);
	 const [dailyMetrics, setDailyMetrics] = useState<DailyMetricsData | null>(null);
	 const [splitInfo, setSplitInfo] = useState<SplitInfoData | null>(null);
	 const [featureWeights, setFeatureWeights] = useState<FeatureWeightsData | null>(null);
	 const [backendBase, setBackendBase] = useState<string>('');
	 const [loading, setLoading] = useState(false);
	 const [error, setError] = useState<string | null>(null);


 const fetchAll = async () => {
   setLoading(true);
   setError(null);
   try {
     const base = await getBackendBaseUrl();
     setBackendBase(base);


     const settled = await Promise.allSettled([
       fetchBackendJson<TrainingAnalyticsData>('/training-analytics'),
       fetchBackendJson<CVMetricsData>('/cv-metrics'),
       fetchBackendJson<SpatialCVData>('/cv-metrics/spatial'),
	       fetchBackendJson<DMTestData>('/dm-test'),
	       fetchBackendJson<DailyMetricsData>('/daily-metrics'),
	       fetchBackendJson<SplitInfoData>('/split-info'),
	       fetchBackendJson<FeatureWeightsData>('/feature-weights'),
	     ]);


	     const [analyticsR, cvR, spatialCvR, dmTestR, dailyR, splitR, featureWeightsR] = settled;


     if (analyticsR.status === 'fulfilled') setTrainingAnalytics(analyticsR.value);
     if (cvR.status === 'fulfilled') setCvMetrics(cvR.value);
     if (spatialCvR.status === 'fulfilled') setSpatialCv(spatialCvR.value);
	     if (dmTestR.status === 'fulfilled') setDmTest(dmTestR.value);
	     if (dailyR.status === 'fulfilled') setDailyMetrics(dailyR.value);
	     if (splitR.status === 'fulfilled') setSplitInfo(splitR.value);
	     if (featureWeightsR.status === 'fulfilled') setFeatureWeights(featureWeightsR.value);


     const primaryFailed = analyticsR.status === 'rejected' && cvR.status === 'rejected';
     if (primaryFailed) {
       const msg = analyticsR.status === 'rejected'
         ? (analyticsR.reason instanceof Error ? analyticsR.reason.message : 'Failed to load analytics')
         : 'Failed to load analytics';
       setError(msg);
     }
   } catch (err) {
     setError(err instanceof Error ? err.message : 'Failed to load global analytics');
   } finally {
     setLoading(false);
   }
 };


 useEffect(() => {
   void fetchAll();
 }, []);


 const mc = trainingAnalytics?.performanceMetricsComparison ?? null;


 const errorMetricsChartData = mc
   ? [
       {
         metric: 'RMSE',
         baseline: mc.baseline.rmse * KWH_TO_J,
         fiAdaBoost: mc.fiAdaBoost.rmse * KWH_TO_J,
       },
       {
         metric: 'MAE',
         baseline: mc.baseline.mae * KWH_TO_J,
         fiAdaBoost: mc.fiAdaBoost.mae * KWH_TO_J,
       },
     ]
   : [];


	 const r2ChartData = mc
	   ? [
	       { metric: 'R²', baseline: mc.baseline.r2 * 100, fiAdaBoost: mc.fiAdaBoost.r2 * 100 },
	     ]
	   : [];


	 const r2ValueSummary = mc
	   ? [
	       { label: 'Baseline AdaBoost', value: mc.baseline.r2, color: 'bg-blue-500' },
	       { label: 'FI-AdaBoost', value: mc.fiAdaBoost.r2, color: 'bg-orange-500' },
	     ]
	   : [];


 const cvChartData =
   cvMetrics?.cv_fold_metrics.map((m) => ({
     fold: `Fold ${m.fold}`,
     baselineRmse: m.baseline_rmse,
     fiRmse: m.fi_rmse,
	     baselineMae: m.baseline_mae,
	     fiMae: m.fi_mae,
	   })) ?? [];


	 const featureWeightChartData =
	   featureWeights?.weights.map((item) => ({
	     rank: item.rank,
	     feature: formatFeatureName(item.feature),
	     rawFeature: item.feature,
	     weight: item.feature_weight,
	     percent: item.feature_weight_percent,
	     source: item.source,
	   })) ?? [];


	 const baselineFeatureWeightChartData =
	   featureWeights?.baseline_weights.map((item) => ({
	     rank: item.rank,
	     feature: formatFeatureName(item.feature),
	     rawFeature: item.feature,
	     weight: item.feature_weight,
	     percent: item.feature_weight_percent,
	     source: item.source,
	   })) ?? [];


 const residualNote = trainingAnalytics
   ? trainingAnalytics.residualSummary.fiStd < trainingAnalytics.residualSummary.baselineStd
     ? 'FI-AdaBoost achieves a tighter residual spread, confirming lower variance in error.'
     : 'The current saved evaluation does not show a tighter FI-AdaBoost residual spread.'
   : null;


 return (
   <div className="space-y-8">
     {/* Banner */}
     <div className="rounded-xl border border-violet-200 bg-violet-50 px-4 py-3 text-sm text-violet-800">
       <span className="font-semibold">Model Analysis</span> — training-wide performance metrics and
       visual diagnostics. These plots are independent of any selected map location.
     </div>


     {error && (
       <Card className="border-2 border-red-200 bg-red-50">
         <CardContent className="pt-6 text-sm text-red-700">{error}</CardContent>
       </Card>
     )}


     {/* ── SECTION 3: Model Comparison ─────────────────────────────────────── */}
     <Card className="border-2 border-violet-200 bg-gradient-to-br from-violet-50 via-fuchsia-50 to-rose-50 shadow-xl">
       <CardHeader>
         <div className="flex items-center justify-between gap-4">
           <div>
             <CardTitle className="text-2xl">Model Comparison</CardTitle>
             <CardDescription>
               Baseline AdaBoost vs FI-AdaBoost — metrics from the saved training run.
             </CardDescription>
           </div>
           <Button variant="outline" size="sm" onClick={fetchAll} disabled={loading} className="gap-2">
             <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
             Refresh
           </Button>
         </div>
       </CardHeader>
       <CardContent className="space-y-8">
         {mc ? (
           <>
             {/* A. Metrics bar chart */}
             <section>
               <h3 className="mb-1 text-lg font-semibold">A. Metrics Comparison</h3>
               <p className="mb-4 text-sm text-slate-500">
                 RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) measure prediction error in J/m²/day — lower values mean more accurate irradiance predictions. R² (coefficient of determination) measures how much of the irradiance variance the model explains; 100% is a perfect fit. Values here are read from the saved <code className="text-xs bg-slate-100 px-1 rounded">results/metrics_summary.csv</code> produced by the last training run.
               </p>
               <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                 <div>
                   <p className="mb-2 text-center text-xs font-medium uppercase tracking-wide text-slate-500">
                     Error Metrics (J/m²/day) — lower is better
                   </p>
                   <ResponsiveContainer width="100%" height={280}>
                     <BarChart data={errorMetricsChartData} barCategoryGap="35%">
                       <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                       <XAxis dataKey="metric" stroke="#64748b" />
                       <YAxis stroke="#64748b" />
                       <Tooltip formatter={(v: number | string) => formatGrouped(Number(v), 6)} />
                       <Legend />
                       <Bar dataKey="baseline" name="Baseline AdaBoost" fill="#3b82f6" radius={[8, 8, 0, 0]}>
                         <LabelList dataKey="baseline" position="top" formatter={(v: number) => formatGrouped(v, 2)} fill="#334155" fontSize={12} />
                       </Bar>
                       <Bar dataKey="fiAdaBoost" name="FI-AdaBoost" fill="#f97316" radius={[8, 8, 0, 0]}>
                         <LabelList dataKey="fiAdaBoost" position="top" formatter={(v: number) => formatGrouped(v, 2)} fill="#334155" fontSize={12} />
                       </Bar>
                     </BarChart>
                   </ResponsiveContainer>
                 </div>
	                 <div>
	                   <p className="mb-2 text-center text-xs font-medium uppercase tracking-wide text-slate-500">
	                     R² Score (%) — higher is better
	                   </p>
	                   <div className="mb-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
	                     {r2ValueSummary.map((item) => (
	                       <div key={item.label} className="flex items-center justify-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm">
	                         <span className={`h-2.5 w-2.5 rounded-full ${item.color}`} />
	                         <span className="font-medium text-slate-600">{item.label}</span>
	                         <span className="font-semibold text-slate-900">{formatR2Percent(item.value)}</span>
	                       </div>
	                     ))}
	                   </div>
	                   <ResponsiveContainer width="100%" height={280}>
	                     <BarChart data={r2ChartData} barCategoryGap="60%" margin={{ top: 12, right: 16, left: 0, bottom: 0 }}>
	                       <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
	                       <XAxis dataKey="metric" stroke="#64748b" />
	                       <YAxis
	                         stroke="#64748b"
	                         domain={([dataMin]: [number, number]) => [Math.max(0, Math.floor(dataMin - 5)), 100]}
                         tickFormatter={(v: number) => `${v}%`}
                       />
	                       <Tooltip formatter={(v: number | string) => formatPercent(Number(v), 6)} />
	                       <Legend />
	                       <Bar dataKey="baseline" name="Baseline AdaBoost" fill="#3b82f6" radius={[8, 8, 0, 0]} />
	                       <Bar dataKey="fiAdaBoost" name="FI-AdaBoost" fill="#f97316" radius={[8, 8, 0, 0]} />
	                     </BarChart>
	                   </ResponsiveContainer>
	                 </div>
               </div>
             </section>


             {/* B. Improvement cards */}
             <section>
               <h3 className="mb-4 text-lg font-semibold">B. Improvement Cards</h3>
               <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                 <ImprovementCard
                   label={mc.rmseImprovementPct >= 0 ? 'RMSE reduced by' : 'RMSE increased by'}
                   value={formatPercent(Math.abs(mc.rmseImprovementPct), 6)}
                   sub="Lower root-mean-square error"
                   positive={mc.rmseImprovementPct >= 0}
                   icon={<TrendingDown className="h-4 w-4" />}
                 />
                 <ImprovementCard
                   label={mc.maeImprovementPct >= 0 ? 'MAE reduced by' : 'MAE increased by'}
                   value={formatPercent(Math.abs(mc.maeImprovementPct), 6)}
                   sub="Lower mean absolute error"
                   positive={mc.maeImprovementPct >= 0}
                   icon={<TrendingDown className="h-4 w-4" />}
                 />
                 <ImprovementCard
                   label={mc.r2ImprovementPct >= 0 ? 'R² increased by' : 'R² decreased by'}
                   value={formatPercent(Math.abs(mc.r2ImprovementPct), 6)}
                   sub="Higher explained variance"
                   positive={mc.r2ImprovementPct >= 0}
                   icon={<TrendingUp className="h-4 w-4" />}
                 />
               </div>
             </section>


             {/* C. Comparison table */}
             <section>
               <h3 className="mb-4 text-lg font-semibold">C. Comparison Table</h3>
               <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
                 <table className="w-full text-sm">
                   <thead className="bg-slate-100 text-slate-700">
                     <tr>
                       <th className="px-4 py-3 text-left font-semibold">Metric</th>
                       <th className="px-4 py-3 text-right font-semibold">Baseline AdaBoost</th>
                       <th className="px-4 py-3 text-right font-semibold">FI-AdaBoost</th>
                       <th className="px-4 py-3 text-right font-semibold">Delta</th>
                     </tr>
                   </thead>
                   <tbody>
                     <tr className="border-t border-slate-200 hover:bg-slate-50">
                       <td className="px-4 py-3 font-medium">RMSE (J/m²/day)</td>
                       <td className="px-4 py-3 text-right">{formatResultsErrorMetric(mc.baseline.rmse)}</td>
                       <td className={`px-4 py-3 text-right font-semibold ${mc.rmseImprovementPct >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                         {formatResultsErrorMetric(mc.fiAdaBoost.rmse)}
                       </td>
                       <td className={`px-4 py-3 text-right ${mc.rmseImprovementPct >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                         {mc.rmseImprovementPct >= 0 ? '↓' : '↑'} {formatPercent(Math.abs(mc.rmseImprovementPct), 6)}
                       </td>
                     </tr>
                     <tr className="border-t border-slate-200 hover:bg-slate-50">
                       <td className="px-4 py-3 font-medium">MAE (J/m²/day)</td>
                       <td className="px-4 py-3 text-right">{formatResultsErrorMetric(mc.baseline.mae)}</td>
                       <td className={`px-4 py-3 text-right font-semibold ${mc.maeImprovementPct >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                         {formatResultsErrorMetric(mc.fiAdaBoost.mae)}
                       </td>
                       <td className={`px-4 py-3 text-right ${mc.maeImprovementPct >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                         {mc.maeImprovementPct >= 0 ? '↓' : '↑'} {formatPercent(Math.abs(mc.maeImprovementPct), 6)}
                       </td>
                     </tr>
                     <tr className="border-t border-slate-200 hover:bg-slate-50">
                       <td className="px-4 py-3 font-medium">R² (%)</td>
                       <td className="px-4 py-3 text-right">{formatR2Percent(mc.baseline.r2)}</td>
                       <td className={`px-4 py-3 text-right font-semibold ${mc.r2ImprovementPct >= 0 ? 'text-sky-700' : 'text-red-700'}`}>
                         {formatR2Percent(mc.fiAdaBoost.r2)}
                       </td>
                       <td className={`px-4 py-3 text-right ${mc.r2ImprovementPct >= 0 ? 'text-sky-700' : 'text-red-700'}`}>
                         {mc.r2ImprovementPct >= 0 ? '↑' : '↓'} {formatPercent(Math.abs(mc.r2ImprovementPct), 6)}
                       </td>
                     </tr>
                   </tbody>
                 </table>
               </div>
             </section>
           </>
         ) : (
           <EmptyState
             message={
               loading
                 ? 'Loading model comparison metrics…'
                 : error ?? 'Metrics unavailable. Start the backend and refresh.'
             }
           />
         )}
       </CardContent>
     </Card>


     {/* ── SECTION 5: Spatial Cross-Validation ─────────────────────────────── */}
     <Card className="border-2 border-indigo-200 bg-gradient-to-br from-indigo-50 via-violet-50 to-purple-50 shadow-xl">
       <CardHeader>
         <CardTitle className="text-2xl">Spatial Cross-Validation (5-Fold)</CardTitle>
         <CardDescription>
           Spatial 5-fold cross-validation randomly partitions the 3,000 coordinates into 5 groups. Each fold trains on 4 groups and validates on the remaining 1. A spatial split was used — rather than a simple random split — because nearby coordinates share atmospheric and urban geometry context; random shuffling would allow information to leak between train and validation sets through spatial autocorrelation.
         </CardDescription>
       </CardHeader>
       <CardContent>
         {spatialCv ? (
           <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
             <table className="w-full text-sm">
               <thead className="bg-slate-100 text-slate-700">
		                 <tr>
		                   <th className="px-4 py-3 text-left font-semibold">Fold</th>
		                   <th className="px-4 py-3 text-right font-semibold">Ada Val RMSE (J/m²/day)</th>
		                   <th className="px-4 py-3 text-right font-semibold">Ada Val MAE (J/m²/day)</th>
		                   <th className="px-4 py-3 text-right font-semibold">Ada Val R² (%)</th>
		                   <th className="px-4 py-3 text-right font-semibold">FI Val RMSE (J/m²/day)</th>
		                   <th className="px-4 py-3 text-right font-semibold">FI Val MAE (J/m²/day)</th>
		                   <th className="px-4 py-3 text-right font-semibold">FI Val R² (%)</th>
	                   <th className="px-4 py-3 text-right font-semibold">RMSE Gain (J/m²/day)</th>
	                   <th className="px-4 py-3 text-right font-semibold">MAE Gain (J/m²/day)</th>
	                 </tr>
	               </thead>
	               <tbody>
	                 {spatialCv.folds.map((f) => {
	                   const rmseGain = f.baseline_val_rmse - f.fi_val_rmse;
	                   const maeGain = f.baseline_val_mae - f.fi_val_mae;
		                   return (
		                     <tr key={f.fold} className="border-t border-slate-200 hover:bg-slate-50">
		                       <td className="px-4 py-3 font-medium">Fold {f.fold}</td>
		                       <td className="px-4 py-3 text-right">{formatExact(f.baseline_val_rmse, 3)}</td>
		                       <td className="px-4 py-3 text-right">{formatExact(f.baseline_val_mae, 3)}</td>
		                       <td className="px-4 py-3 text-right">{formatR2Percent(f.baseline_val_r2, 4)}</td>
		                       <td className="px-4 py-3 text-right font-semibold text-orange-600">{formatExact(f.fi_val_rmse, 3)}</td>
		                       <td className="px-4 py-3 text-right font-semibold text-orange-600">{formatExact(f.fi_val_mae, 3)}</td>
		                       <td className="px-4 py-3 text-right font-semibold text-emerald-600">{formatR2Percent(f.fi_val_r2, 4)}</td>
	                       <td className={`px-4 py-3 text-right ${rmseGain >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
	                         {rmseGain >= 0 ? '↓' : '↑'} {formatExact(Math.abs(rmseGain), 3)}
	                       </td>
	                       <td className={`px-4 py-3 text-right ${maeGain >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
	                         {maeGain >= 0 ? '↓' : '↑'} {formatExact(Math.abs(maeGain), 3)}
	                       </td>
	                     </tr>
	                   );
	                 })}
		                 <tr className="border-t-2 border-slate-300 bg-slate-50 font-semibold">
		                   <td className="px-4 py-3">Average</td>
		                   <td className="px-4 py-3 text-right">{formatExact(spatialCv.average.baseline_val_rmse, 3)}</td>
		                   <td className="px-4 py-3 text-right">{formatExact(spatialCv.average.baseline_val_mae, 3)}</td>
		                   <td className="px-4 py-3 text-right">{formatR2Percent(spatialCv.average.baseline_val_r2, 4)}</td>
		                   <td className="px-4 py-3 text-right text-orange-600">{formatExact(spatialCv.average.fi_val_rmse, 3)}</td>
		                   <td className="px-4 py-3 text-right text-orange-600">{formatExact(spatialCv.average.fi_val_mae, 3)}</td>
		                   <td className="px-4 py-3 text-right text-emerald-600">{formatR2Percent(spatialCv.average.fi_val_r2, 4)}</td>
	                   <td className={`px-4 py-3 text-right ${spatialCv.average.baseline_val_rmse - spatialCv.average.fi_val_rmse >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
	                     {spatialCv.average.baseline_val_rmse - spatialCv.average.fi_val_rmse >= 0 ? '↓' : '↑'} {formatExact(Math.abs(spatialCv.average.baseline_val_rmse - spatialCv.average.fi_val_rmse), 3)}
	                   </td>
	                   <td className={`px-4 py-3 text-right ${spatialCv.average.baseline_val_mae - spatialCv.average.fi_val_mae >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
	                     {spatialCv.average.baseline_val_mae - spatialCv.average.fi_val_mae >= 0 ? '↓' : '↑'} {formatExact(Math.abs(spatialCv.average.baseline_val_mae - spatialCv.average.fi_val_mae), 3)}
	                   </td>
	                 </tr>
               </tbody>
             </table>
           </div>
         ) : (
           <EmptyState message={loading ? 'Loading spatial CV…' : 'Spatial CV metrics unavailable.'} />
         )}
	       </CardContent>
	     </Card>


	     {/* ── SECTION 6: Feature Weight Proof ─────────────────────────────────── */}
	     <Card className="border-2 border-orange-200 bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50 shadow-xl">
	       <CardHeader>
	         <CardTitle className="text-2xl">Feature Weight Importance</CardTitle>
	         <CardDescription>
	           Ranked actual fitted baseline AdaBoost weights and FI-AdaBoost weights served from <code>results/</code>.
	         </CardDescription>
	       </CardHeader>
	       <CardContent>
	         {featureWeights ? (
	           <div className="space-y-8">
	             <section>
	               <div className="mb-3">
	                 <h3 className="text-lg font-semibold">Baseline AdaBoost Feature Weight Importance</h3>
	                 <p className="text-sm text-slate-500">
	                   Actual fitted AdaBoost feature weights from <code>results/{featureWeights.baseline_source_file}</code>.
	                 </p>
	               </div>
	               <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
	                 <ResponsiveContainer width="100%" height={380}>
	                   <BarChart
	                     data={baselineFeatureWeightChartData}
	                     layout="vertical"
	                     margin={{ top: 12, right: 84, left: 12, bottom: 12 }}
	                   >
	                     <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
	                     <XAxis
	                       type="number"
	                       stroke="#64748b"
	                       tickFormatter={(v: number) => formatPercent(v * 100, 0)}
	                       domain={[0, (dataMax: number) => Math.max(dataMax * 1.15, 0.05)]}
	                     />
	                     <YAxis
	                       dataKey="feature"
	                       type="category"
	                       stroke="#64748b"
	                       width={170}
	                       tick={{ fontSize: 11 }}
	                     />
	                     <Tooltip
	                       formatter={(v: number | string) => formatSmallPercent(Number(v) * 100, 6)}
	                       labelFormatter={(label: string | number) => String(label)}
	                     />
	                     <Bar dataKey="weight" name="Baseline feature weight" fill="#3b82f6" radius={[0, 6, 6, 0]}>
	                       <LabelList dataKey="percent" position="right" formatter={(v: number) => formatSmallPercent(v, 4)} fill="#334155" fontSize={12} />
	                     </Bar>
	                   </BarChart>
	                 </ResponsiveContainer>
	               </div>
	             </section>
	             <section>
	               <div className="mb-3">
	                 <h3 className="text-lg font-semibold">FI-AdaBoost Feature Weight Importance</h3>
	                 <p className="text-sm text-slate-500">
	                   FI-AdaBoost feature-aware weights from <code>results/{featureWeights.source_file}</code>.
	                 </p>
	               </div>
	               <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
	                 <ResponsiveContainer width="100%" height={380}>
	                   <BarChart
	                     data={featureWeightChartData}
	                     layout="vertical"
	                     margin={{ top: 12, right: 84, left: 12, bottom: 12 }}
	                   >
	                     <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
	                     <XAxis
	                       type="number"
	                       stroke="#64748b"
	                       tickFormatter={(v: number) => formatPercent(v * 100, 0)}
	                       domain={[0, (dataMax: number) => Math.max(dataMax * 1.15, 0.05)]}
	                     />
	                     <YAxis
	                       dataKey="feature"
	                       type="category"
	                       stroke="#64748b"
	                       width={170}
	                       tick={{ fontSize: 11 }}
	                     />
	                     <Tooltip
	                       formatter={(v: number | string) => formatSmallPercent(Number(v) * 100, 6)}
	                       labelFormatter={(label: string | number) => String(label)}
	                     />
	                     <Bar dataKey="weight" name="FI feature weight" fill="#f97316" radius={[0, 6, 6, 0]}>
	                       <LabelList dataKey="percent" position="right" formatter={(v: number) => formatSmallPercent(v, 4)} fill="#334155" fontSize={12} />
	                     </Bar>
	                   </BarChart>
	                 </ResponsiveContainer>
	               </div>
	             </section>
	           </div>
	         ) : (
	           <EmptyState message={loading ? 'Loading feature weights…' : 'Feature weights unavailable.'} />
	         )}
	       </CardContent>
	     </Card>


	     {/* ── SECTION 6: Diebold-Mariano Test ─────────────────────────────────── */}
	     <Card className="border-2 border-emerald-200 bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50 shadow-xl">
       <CardHeader>
         <CardTitle className="text-2xl">Diebold-Mariano Statistical Test</CardTitle>
         <CardDescription>
           The Diebold-Mariano test (Harvey et al., 1997) is a formal hypothesis test for equal predictive accuracy. H₀: both models have the same forecast error. A <strong>positive DM statistic</strong> means FI-AdaBoost produces smaller errors than the baseline. A p-value below 0.05 rejects H₀ — the accuracy difference is statistically significant, not due to chance. The variance estimate uses Newey-West HAC corrections for error autocorrelation.
         </CardDescription>
       </CardHeader>
       <CardContent>
         {dmTest ? (
           <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
             <table className="w-full text-sm">
               <thead className="bg-slate-100 text-slate-700">
                 <tr>
                   <th className="px-4 py-3 text-left font-semibold">Domain</th>
                   <th className="px-4 py-3 text-right font-semibold">DM Statistic (unitless)</th>
                   <th className="px-4 py-3 text-right font-semibold">p-value (unitless)</th>
                   <th className="px-4 py-3 text-right font-semibold">n (samples)</th>
                   <th className="px-4 py-3 text-left font-semibold">Significant?</th>
                   <th className="px-4 py-3 text-left font-semibold">Interpretation</th>
                 </tr>
               </thead>
               <tbody>
                 {([['Spatial', dmTest.spatial], ['Daily', dmTest.daily]] as const).map(([label, r]) => (
                   <tr key={label} className="border-t border-slate-200 hover:bg-slate-50">
                     <td className="px-4 py-3 font-medium">{label}</td>
                     <td className="px-4 py-3 text-right font-mono">{formatExact(r.dm_statistic, 4)}</td>
                     <td className="px-4 py-3 text-right font-mono">{r.p_value < 0.0001 ? '< 0.0001' : formatExact(r.p_value, 5)}</td>
                     <td className="px-4 py-3 text-right">{r.n_samples}</td>
                     <td className="px-4 py-3">
                       <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${r.significant ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-600'}`}>
                         {r.significant ? '✓ Yes (p < 0.05)' : '✗ No'}
                       </span>
                     </td>
                     <td className="px-4 py-3 text-xs text-slate-600">{r.interpretation}</td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
         ) : (
           <EmptyState message={loading ? 'Loading DM test results…' : 'DM test results unavailable.'} />
         )}
       </CardContent>
     </Card>


     {/* ── SECTION 7: Daily Temporal Metrics ───────────────────────────────── */}
     <Card className="border-2 border-amber-200 bg-gradient-to-br from-amber-50 via-yellow-50 to-orange-50 shadow-xl">
       <CardHeader>
         <CardTitle className="text-2xl">Daily Temporal Split Metrics</CardTitle>
         <CardDescription>
           An 80/20 chronological split on the 365-day NASA POWER centroid time series for Davao City. The test set is strictly future dates — the model cannot see any temporal context from the test period. This is a harder evaluation than spatial cross-validation because there is no spatial autocorrelation to exploit; the model must generalise to unseen time.
         </CardDescription>
       </CardHeader>
       <CardContent>
         {dailyMetrics ? (
           <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
             <table className="w-full text-sm">
               <thead className="bg-slate-100 text-slate-700">
                 <tr>
                   <th className="px-4 py-3 text-left font-semibold">Model</th>
                   <th className="px-4 py-3 text-right font-semibold">RMSE (J/day)</th>
                   <th className="px-4 py-3 text-right font-semibold">MAE (J/day)</th>
                   <th className="px-4 py-3 text-right font-semibold">R² (%)</th>
                   <th className="px-4 py-3 text-left font-semibold">Split</th>
                 </tr>
               </thead>
               <tbody>
                 {dailyMetrics.results.map((r, i) => (
                   <tr key={i} className="border-t border-slate-200 hover:bg-slate-50">
                     <td className="px-4 py-3 font-medium">{r.model}</td>
                     <td className="px-4 py-3 text-right">{r.rmse_j.toLocaleString('en-US', { maximumFractionDigits: 2 })}</td>
                     <td className="px-4 py-3 text-right">{r.mae_j.toLocaleString('en-US', { maximumFractionDigits: 2 })}</td>
                     <td className="px-4 py-3 text-right">{formatR2Percent(r.r2)}</td>
                     <td className="px-4 py-3 text-xs text-slate-500">{r.split}</td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
         ) : (
           <EmptyState message={loading ? 'Loading daily metrics…' : 'Daily metrics unavailable.'} />
         )}
       </CardContent>
     </Card>


     {/* ── SECTION 9: Research Plots ────────────────────────────────────────── */}
     <Card className="border-2 border-rose-200 bg-gradient-to-br from-rose-50 via-pink-50 to-fuchsia-50 shadow-xl">
       <CardHeader>
         <CardTitle className="text-2xl">Research Plots from Training Run</CardTitle>
         <CardDescription>
           Publication-quality figures generated by the last training run and served from <code>results/</code>. Each plot is reproducible by re-running <code>model_training.py</code>; the images update automatically after the next training run.
         </CardDescription>
       </CardHeader>
       <CardContent>
         {backendBase ? (
           <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
             {[
               {
                 file: 'metrics_comparison.png',
                 caption: 'Metrics Comparison',
                 description: 'Side-by-side bar chart of RMSE, MAE, and R² for both models on the held-out test set. Lower RMSE and MAE, and higher R², indicate better performance.',
               },
               {
                 file: 'actual_vs_predicted.png',
                 caption: 'Actual vs Predicted',
                 description: 'Scatter plot of predicted vs actual effective GHI values. Points clustered along the dashed diagonal indicate accurate predictions; deviation signals systematic error.',
               },
               {
                 file: 'residuals.png',
                 caption: 'Residual Analysis',
                 description: 'Residual plot (actual − predicted) vs predicted value. A horizontal band centred near zero indicates unbiased predictions; patterns or fanning suggest model misfit.',
               },
               {
                 file: 'overfit_check.png',
                 caption: 'Overfit Check (Train vs Test)',
                 description: 'Train vs test RMSE per model. A large gap between the two bars signals overfitting — the model memorised training data but generalises poorly.',
               },
	               {
	                 file: 'baseline_feature_weight_importance.png',
	                 caption: 'Baseline AdaBoost Feature Weight Importance',
	                 description: 'Actual fitted baseline AdaBoost feature weights exported from the training run for direct comparison against FI-AdaBoost feature-aware weights.',
	               },
	               {
	                 file: 'feature_weight_importance.png',
	                 caption: 'FI-AdaBoost Feature Weight Importance',
	                 description: 'Explicit feature-weight proof artifact showing the FI-AdaBoost weights used by the feature-aware boosting mechanism.',
	               },
	               {
	                 file: 'energy_distribution.png',
	                 caption: 'Energy Distribution',
                 description: 'Per-building annual solar energy yield distribution across all 3,000 Davao City rooftops. The dashed line marks the mean yield; the spread reflects rooftop geometry and orientation diversity.',
               },
               {
                 file: 'total_energy_comparison.png',
                 caption: 'Total Energy Comparison',
                 description: 'Aggregate rooftop solar energy potential (kWh/year) for Davao City by model. The difference between bars quantifies how the feature-importance weighting shifts the city-wide energy estimate.',
               },
             ].map(({ file, caption, description }) => (
               <div key={file} className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                 <img
                   src={`${backendBase}/results/images/${file}`}
                   alt={caption}
                   className="w-full object-contain"
                   loading="lazy"
                 />
                 <div className="px-3 py-3 border-t border-slate-100">
                   <p className="text-center text-xs font-semibold text-slate-700">{caption}</p>
                   <p className="mt-1 text-center text-xs text-slate-500 leading-relaxed">{description}</p>
                 </div>
               </div>
             ))}
           </div>
         ) : (
           <EmptyState message={loading ? 'Resolving backend URL…' : 'Backend URL unavailable — cannot load research plots.'} />
         )}
       </CardContent>
     </Card>
   </div>
 );
}


// ─── Sub-components ───────────────────────────────────────────────────────────


interface ImprovementCardProps {
 label: string;
 value: string;
 sub: string;
 positive: boolean;
 icon: React.ReactNode;
}


function ImprovementCard({ label, value, sub, positive, icon }: ImprovementCardProps) {
 const border = positive ? 'border-emerald-200' : 'border-red-200';
 const bg    = positive ? 'bg-emerald-50'     : 'bg-red-50';
 const color = positive ? 'text-emerald-700'  : 'text-red-700';


 return (
   <Card className={`border-2 ${border} ${bg}`}>
     <CardHeader className="pb-3">
       <CardTitle className={`flex items-center gap-2 text-sm ${color}`}>
         {icon}
         {label}
       </CardTitle>
     </CardHeader>
     <CardContent>
       <p className={`text-2xl font-semibold ${color}`}>{value}</p>
       <p className="mt-1 text-xs text-slate-500">{sub}</p>
     </CardContent>
   </Card>
 );
}


interface ScatterPanelProps {
 title: string;
 data: PredictedActualPoint[];
 color: string;
 domainMin: number;
 domainMax: number;
}


function ScatterPanel({ title, data, color, domainMin, domainMax }: ScatterPanelProps) {
 return (
   <Card className="border border-slate-200 bg-white shadow-sm">
     <CardHeader className="pb-3">
       <CardTitle className="text-base">{title}</CardTitle>
       <CardDescription>Predicted (y-axis) vs Actual (x-axis)</CardDescription>
     </CardHeader>
     <CardContent>
       <ResponsiveContainer width="100%" height={300}>
         <ScatterChart>
           <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
           <XAxis
             type="number"
             dataKey="actual"
             name="Actual"
             domain={[domainMin, domainMax]}
             stroke="#64748b"
             label={{ value: 'Actual', position: 'insideBottom', offset: -5, fontSize: 12 }}
           />
           <YAxis
             type="number"
             dataKey="predicted"
             name="Predicted"
             domain={[domainMin, domainMax]}
             stroke="#64748b"
             label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fontSize: 12 }}
           />
           <Tooltip formatter={(v: number | string) => Number(v).toFixed(2)} />
           <ReferenceLine
             segment={[
               { x: domainMin, y: domainMin },
               { x: domainMax, y: domainMax },
             ]}
             stroke="#0f172a"
             strokeDasharray="4 4"
           />
           <Scatter data={data} fill={color} fillOpacity={0.7} />
         </ScatterChart>
       </ResponsiveContainer>
     </CardContent>
   </Card>
 );
}


function EmptyState({ message }: { message: string }) {
 return (
   <div className="rounded-xl border border-dashed border-slate-300 bg-white/70 p-8 text-center text-sm text-slate-500">
     {message}
   </div>
 );
}
