import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
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
import { fetchBackendJson } from '../lib/backend';

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

// ─── Main component ───────────────────────────────────────────────────────────

export function GlobalModelAnalytics() {
  const [trainingAnalytics, setTrainingAnalytics] = useState<TrainingAnalyticsData | null>(null);
  const [cvMetrics, setCvMetrics] = useState<CVMetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const [analytics, cv] = await Promise.all([
        fetchBackendJson<TrainingAnalyticsData>('/training-analytics'),
        fetchBackendJson<CVMetricsData>('/cv-metrics'),
      ]);
      setTrainingAnalytics(analytics);
      setCvMetrics(cv);
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
        { metric: 'RMSE', baseline: mc.baseline.rmse, fiAdaBoost: mc.fiAdaBoost.rmse },
        { metric: 'MAE',  baseline: mc.baseline.mae,  fiAdaBoost: mc.fiAdaBoost.mae  },
      ]
    : [];

  const r2ChartData = mc
    ? [
        { metric: 'R²', baseline: mc.baseline.r2 * 100, fiAdaBoost: mc.fiAdaBoost.r2 * 100 },
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
                  Error metrics (lower is better) and R² as percentage (higher is better). Shown on separate axes so R² isn't dwarfed by RMSE/MAE.
                </p>
                <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                  <div>
                    <p className="mb-2 text-center text-xs font-medium uppercase tracking-wide text-slate-500">
                      Error Metrics — lower is better
                    </p>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={errorMetricsChartData} barCategoryGap="35%">
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="metric" stroke="#64748b" />
                        <YAxis stroke="#64748b" />
                        <Tooltip formatter={(v: number | string) => Number(v).toFixed(2)} />
                        <Legend />
                        <Bar dataKey="baseline" name="Baseline AdaBoost" fill="#94a3b8" radius={[8, 8, 0, 0]} />
                        <Bar dataKey="fiAdaBoost" name="FI-AdaBoost" fill="#f97316" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <p className="mb-2 text-center text-xs font-medium uppercase tracking-wide text-slate-500">
                      R² Score (%) — higher is better
                    </p>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={r2ChartData} barCategoryGap="60%">
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="metric" stroke="#64748b" />
                        <YAxis
                          stroke="#64748b"
                          domain={[85, 100]}
                          tickFormatter={(v: number) => `${v}%`}
                        />
                        <Tooltip formatter={(v: number | string) => `${Number(v).toFixed(2)}%`} />
                        <Legend />
                        <Bar dataKey="baseline" name="Baseline AdaBoost" fill="#94a3b8" radius={[8, 8, 0, 0]} />
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
                    label="RMSE reduced by"
                    value={`${mc.rmseImprovementPct.toFixed(1)}%`}
                    sub="Lower root-mean-square error"
                    positive
                    icon={<TrendingDown className="h-4 w-4" />}
                  />
                  <ImprovementCard
                    label="MAE reduced by"
                    value={`${mc.maeImprovementPct.toFixed(1)}%`}
                    sub="Lower mean absolute error"
                    positive
                    icon={<TrendingDown className="h-4 w-4" />}
                  />
                  <ImprovementCard
                    label="R² increased by"
                    value={`${mc.r2ImprovementPct.toFixed(2)}%`}
                    sub="Higher explained variance"
                    positive
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
                        <td className="px-4 py-3 font-medium">RMSE</td>
                        <td className="px-4 py-3 text-right">{mc.baseline.rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right font-semibold text-emerald-700">{mc.fiAdaBoost.rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-emerald-700">↓ {mc.rmseImprovementPct.toFixed(1)}%</td>
                      </tr>
                      <tr className="border-t border-slate-200 hover:bg-slate-50">
                        <td className="px-4 py-3 font-medium">MAE</td>
                        <td className="px-4 py-3 text-right">{mc.baseline.mae.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right font-semibold text-emerald-700">{mc.fiAdaBoost.mae.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-emerald-700">↓ {mc.maeImprovementPct.toFixed(1)}%</td>
                      </tr>
                      <tr className="border-t border-slate-200 hover:bg-slate-50">
                        <td className="px-4 py-3 font-medium">R²</td>
                        <td className="px-4 py-3 text-right">{(mc.baseline.r2 * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 text-right font-semibold text-sky-700">{(mc.fiAdaBoost.r2 * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 text-right text-sky-700">↑ {mc.r2ImprovementPct.toFixed(2)}%</td>
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

      {/* ── SECTION 4: Visual Analysis ───────────────────────────────────────── */}
      <Card className="border-2 border-sky-200 bg-gradient-to-br from-sky-50 via-cyan-50 to-teal-50 shadow-xl">
        <CardHeader>
          <CardTitle className="text-2xl">Visual Analysis</CardTitle>
          <CardDescription>
            Predicted vs Actual scatter, error distribution histogram, and cross-validation fold chart.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-10">
          {/* A. Predicted vs Actual */}
          {/* <section>
            <h3 className="mb-1 text-lg font-semibold">A. Predicted vs Actual</h3>
            <p className="mb-4 text-sm text-slate-500">
              Points on — or close to — the dashed diagonal indicate accurate predictions.
            </p>
            {trainingAnalytics ? (
              <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <ScatterPanel
                  title="Baseline AdaBoost"
                  data={trainingAnalytics.predictedVsActual.baseline}
                  color="#94a3b8"
                  domainMin={trainingAnalytics.predictedVsActual.domainMin}
                  domainMax={trainingAnalytics.predictedVsActual.domainMax}
                />
                <ScatterPanel
                  title="FI-AdaBoost"
                  data={trainingAnalytics.predictedVsActual.fiAdaBoost}
                  color="#f97316"
                  domainMin={trainingAnalytics.predictedVsActual.domainMin}
                  domainMax={trainingAnalytics.predictedVsActual.domainMax}
                />
              </div>
            ) : (
              <EmptyState
                message={loading ? 'Loading scatter plots…' : error ?? 'Training analytics unavailable.'}
              />
            )}
          </section> */}

          {/* B. Error distribution */}
          <section>
            <h3 className="mb-1 text-lg font-semibold">B. Error Distribution</h3>
            <p className="mb-4 text-sm text-slate-500">
              Residual histogram — FI-AdaBoost should show a tighter spread around zero.
            </p>
            {trainingAnalytics ? (
              <>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={trainingAnalytics.errorDistribution} barCategoryGap="10%">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="bucket" stroke="#64748b" interval={2} angle={-25} textAnchor="end" height={70} />
                    <YAxis stroke="#64748b" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="baseline" name="Baseline residual count" fill="#94a3b8" radius={[5, 5, 0, 0]} />
                    <Bar dataKey="fiAdaBoost" name="FI-AdaBoost residual count" fill="#f97316" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                {residualNote && (
                  <div className="mt-4 rounded-xl border border-sky-200 bg-white p-4 text-sm text-slate-700 shadow-sm">
                    <p>{residualNote}</p>
                    <p className="mt-2 text-xs text-slate-500">
                      Residual std — Baseline: {trainingAnalytics.residualSummary.baselineStd.toFixed(2)},
                      FI-AdaBoost: {trainingAnalytics.residualSummary.fiStd.toFixed(2)}.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <EmptyState
                message={loading ? 'Loading error distribution…' : error ?? 'Training analytics unavailable.'}
              />
            )}
          </section>

          {/* C. Cross-validation */}
          <section>
            <h3 className="mb-1 text-lg font-semibold">C. Cross-validation Fold Chart</h3>
            <p className="mb-4 text-sm text-slate-500">
              RMSE per fold (5-fold time-series split). FI-AdaBoost should consistently track below Baseline.
            </p>
            {cvChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={cvChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="fold" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip formatter={(v: number | string) => Number(v).toFixed(2)} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="baselineRmse"
                    name="Baseline RMSE"
                    stroke="#94a3b8"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                    activeDot={{ r: 7 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="fiRmse"
                    name="FI-AdaBoost RMSE"
                    stroke="#f97316"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                    activeDot={{ r: 7 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                message={loading ? 'Loading cross-validation chart…' : error ?? 'CV metrics unavailable.'}
              />
            )}

            {/* CV summary table */}
            {cvMetrics && (
              <div className="mt-6 overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
                <table className="w-full text-sm">
                  <thead className="bg-slate-100 text-slate-700">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold">Fold</th>
                      <th className="px-4 py-3 text-right font-semibold">Baseline RMSE</th>
                      <th className="px-4 py-3 text-right font-semibold">FI RMSE</th>
                      <th className="px-4 py-3 text-right font-semibold">Baseline MAE</th>
                      <th className="px-4 py-3 text-right font-semibold">FI MAE</th>
                      <th className="px-4 py-3 text-right font-semibold">Baseline R²</th>
                      <th className="px-4 py-3 text-right font-semibold">FI R²</th>
                      <th className="px-4 py-3 text-right font-semibold">R² Gain</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cvMetrics.cv_fold_metrics.map((m) => (
                      <tr key={m.fold} className="border-t border-slate-200 hover:bg-slate-50">
                        <td className="px-4 py-3 font-medium">Fold {m.fold}</td>
                        <td className="px-4 py-3 text-right">{m.baseline_rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right font-semibold text-orange-600">{m.fi_rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right">{m.baseline_mae.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right font-semibold text-orange-600">{m.fi_mae.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right">{(m.baseline_r2 * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 text-right font-semibold text-emerald-600">{(m.fi_r2 * 100).toFixed(2)}%</td>
                        <td className="px-4 py-3 text-right text-emerald-600">+{((m.fi_r2 - m.baseline_r2) * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                    {/* Average row */}
                    <tr className="border-t-2 border-slate-300 bg-slate-50 font-semibold">
                      <td className="px-4 py-3">Average</td>
                      <td className="px-4 py-3 text-right">{cvMetrics.average_metrics.baseline.rmse.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right text-orange-600">{cvMetrics.average_metrics.fiAdaBoost.rmse.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right">{cvMetrics.average_metrics.baseline.mae.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right text-orange-600">{cvMetrics.average_metrics.fiAdaBoost.mae.toFixed(2)}</td>
                      <td className="px-4 py-3 text-right">{(cvMetrics.average_metrics.baseline.r2 * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right text-emerald-600">{(cvMetrics.average_metrics.fiAdaBoost.r2 * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right text-emerald-600">+{((cvMetrics.average_metrics.fiAdaBoost.r2 - cvMetrics.average_metrics.baseline.r2) * 100).toFixed(2)}%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </section>
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
