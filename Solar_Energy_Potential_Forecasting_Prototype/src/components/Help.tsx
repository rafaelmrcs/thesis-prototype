import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import {
  Sun,
  Map,
  BarChart3,
  FlaskConical,
  Database,
  Layers,
  BookOpen,
  GitBranch,
  Zap,
  Target,
  Cpu,
  TrendingUp,
  AlertTriangle,
} from 'lucide-react';

export function Help() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Hero */}
      <div className="rounded-2xl bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50 border-2 border-orange-200 p-6 shadow-lg">
        <div className="flex items-start gap-4">
          <div className="w-14 h-14 bg-gradient-to-br from-yellow-400 via-orange-500 to-red-500 rounded-xl flex items-center justify-center shadow-lg shrink-0">
            <Sun className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Solar Energy Potential Forecasting</h1>
            <p className="text-gray-600 mt-1">
              A thesis prototype for estimating rooftop solar energy potential across <strong>Davao City, Philippines</strong> using a proposed <strong>Feature-Importance AdaBoost (FI-AdaBoost)</strong> regression model compared against a standard AdaBoost baseline.
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              <Badge className="bg-orange-100 text-orange-700 border border-orange-200">Thesis Prototype</Badge>
              <Badge className="bg-blue-100 text-blue-700 border border-blue-200">NASA POWER + OpenStreetMap</Badge>
              <Badge className="bg-violet-100 text-violet-700 border border-violet-200">University of Mindanao</Badge>
            </div>
          </div>
        </div>
      </div>

      {/* About the study */}
      <Card className="border-2 border-blue-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <BookOpen className="w-5 h-5 text-blue-600" />
            About This Study
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <p>
            This study evaluates whether a <strong>Feature-Importance-Aware AdaBoost (FI-AdaBoost)</strong> regression model can improve solar irradiance forecasting for rooftop suitability analysis compared to a standard AdaBoost baseline. The research focuses on <strong>Davao City, Philippines</strong> as the study area.
          </p>
          <p>
            The pipeline samples 3,000 random coordinates within Davao City, fetches annual solar irradiance from <strong>NASA POWER</strong>, derives rooftop features from <strong>OpenStreetMap</strong> building footprints, and trains both models on the same target: <em>pvlib POA-adjusted effective irradiance (J/m²/day)</em>.
          </p>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 font-mono text-xs text-slate-700 space-y-1">
            <p>Solar Energy Potential (kWh/day)  = Predicted GHI (kWh/m²/day) × Rooftop Area (m²)</p>
            <p>Solar Energy Potential (kWh/year) = Daily SEP × 365</p>
          </div>
          <p className="text-xs text-gray-500 italic">
            Panel efficiency is intentionally excluded — SEP represents solar resource availability, not installed PV output.
          </p>
        </CardContent>
      </Card>

      {/* Two experiments */}
      <Card className="border-2 border-violet-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <GitBranch className="w-5 h-5 text-violet-600" />
            Two Experiments
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <p>The study runs two comparison experiments, each on a separate code branch:</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded">main</span>
                <p className="font-semibold text-gray-800">Experiment 1 — Algorithm Comparison</p>
              </div>
              <p className="text-xs text-gray-600">
                Both Baseline AdaBoost and FI-AdaBoost use the <strong>same 8 features</strong>. The only difference is the boosting algorithm. This isolates the algorithmic contribution of FI-AdaBoost.
              </p>
              <div className="text-xs text-gray-500 space-y-0.5">
                <p>Baseline RMSE: <strong>4,870.80 J/m²/day</strong></p>
                <p>FI-AdaBoost RMSE: <strong>3,375.91 J/m²/day</strong></p>
                <p>Improvement: <strong>30.7%</strong> &nbsp;·&nbsp; DM = 14.74 (p &lt; 0.001)</p>
              </div>
            </div>
            <div className="rounded-xl border border-orange-200 bg-orange-50 p-4 shadow-sm space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono bg-orange-100 text-orange-600 px-2 py-0.5 rounded">mainbaseline</span>
                <p className="font-semibold text-orange-900">Experiment 2 — Full System Comparison</p>
              </div>
              <p className="text-xs text-orange-800">
                Baseline uses only <strong>2 features</strong> (lat, lon). FI-AdaBoost uses all <strong>8 features</strong>. This shows the practical advantage of the full proposed system.
              </p>
              <div className="text-xs text-orange-700 space-y-0.5">
                <p>Baseline RMSE: <strong>28,144.27 J/m²/day</strong></p>
                <p>FI-AdaBoost RMSE: <strong>6,682.48 J/m²/day</strong></p>
                <p>Improvement: <strong>76.3%</strong> &nbsp;·&nbsp; DM = 10.048 (p &lt; 0.001)</p>
              </div>
            </div>
          </div>
          <p className="text-xs text-gray-500">
            Both experiments share the same daily temporal evaluation (365-day NASA POWER centroid data) — no statistically significant difference was found on the daily dataset (DM = −1.54, p = 0.124).
          </p>
        </CardContent>
      </Card>

      {/* How FI-AdaBoost differs */}
      <Card className="border-2 border-violet-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Cpu className="w-5 h-5 text-violet-600" />
            How FI-AdaBoost Differs from Baseline AdaBoost
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <p>
            Standard AdaBoost reweights training samples at each boosting round based solely on prediction error — samples with large errors receive higher weight so subsequent trees focus on them. This treats all features as equally informative when deciding where to concentrate learning.
          </p>
          <p>
            FI-AdaBoost adds a <strong>feature-engagement score Φᵢ</strong> (Phi sub-i) at each round. After the current tree is fitted, its normalised Gini importances (summing to 1) are used as weights over the features. For each sample, Φᵢ is the weighted sum of its scaled absolute feature values — a score in [0, 1] reflecting how strongly that sample activates the features the current tree found important.
          </p>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 font-mono text-xs text-slate-700 space-y-1">
            <p>Φᵢ  = Σⱼ φⱼ · |xᵢⱼ_scaled|   (per-sample engagement score)</p>
            <p>lossᵢ = errorᵢ × Φᵢ            (feature-importance-modulated loss)</p>
          </div>
          <p>
            This modulated loss drives both the error rate εₜ and the sample weight update, steering subsequent trees toward hard-to-predict samples in <em>feature-important regions</em> of the input space — not just anywhere the error is large. Final prediction aggregates all weak-learner outputs using a <strong>weighted median</strong> (not mean), which is more robust to outlier trees.
          </p>
        </CardContent>
      </Card>

      {/* Prediction Pipeline */}
      <Card className="border-2 border-green-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Zap className="w-5 h-5 text-green-600" />
            How the Prediction Works
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <ol className="space-y-2 list-decimal list-inside">
            <li><strong>Pin location</strong> → Overpass API fetches the nearest OSM building polygon</li>
            <li><strong>Building features:</strong> azimuth (longest-edge bearing), rooftop area (m²), orientation_score (south-facing penalty), shading_factor (area-ratio obstruction proxy), SEI_norm (Solar Exposure Index ÷ training max)</li>
            <li><strong>NASA POWER</strong> rolling window: GHI, clearness index (KT), air temp (T2M), rel. humidity (RH2M)</li>
            <li><strong>pvlib Ineichen</strong> clear-sky model → ghi_clear_annual + sunshine_hours → <code className="text-xs bg-slate-100 px-1 rounded">clear_sky_ratio = GHI ÷ ghi_clear</code></li>
            <li>8-feature vector: <code className="text-xs bg-slate-100 px-1 rounded">[lat, lon, azimuth, orientation_score, shading_factor, SEI_norm, clear_sky_ratio, sunshine_hours]</code></li>
            <li><strong>FI-AdaBoost</strong> predicts GHI (kWh/m²/day) → Solar Energy Potential computed from rooftop area</li>
          </ol>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 font-mono text-xs text-slate-700 space-y-1">
            <p>SEP (kWh/day)  = GHI (kWh/m²/day) × Rooftop Area (m²)</p>
            <p>SEP (kWh/year) = Daily SEP × 365</p>
          </div>
        </CardContent>
      </Card>

      {/* Reading the Results */}
      <Card className="border-2 border-orange-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Target className="w-5 h-5 text-orange-600" />
            Reading the Results
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm text-gray-700">
          <div className="space-y-3">
            <p className="font-semibold text-gray-800">Irradiance Resource Level</p>
            <p className="text-xs text-gray-600">Percentage of the 7 kWh/m²/day practical maximum:</p>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-2 font-mono text-xs text-slate-700 text-center">
              (GHI ÷ 7) × 100 %
            </div>
            <div className="overflow-hidden rounded-lg border border-slate-200 text-xs">
              <table className="w-full">
                <thead className="bg-slate-100 text-slate-700">
                  <tr>
                    <th className="px-3 py-2 text-left font-semibold">Rating</th>
                    <th className="px-3 py-2 text-left font-semibold">Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-slate-100"><td className="px-3 py-1.5 font-medium text-emerald-700">Excellent</td><td className="px-3 py-1.5 text-slate-600">≥ 5.5 kWh/m²/day</td></tr>
                  <tr className="border-t border-slate-100"><td className="px-3 py-1.5 font-medium text-blue-700">Very Good</td><td className="px-3 py-1.5 text-slate-600">≥ 4.5 kWh/m²/day</td></tr>
                  <tr className="border-t border-slate-100"><td className="px-3 py-1.5 font-medium text-amber-700">Good</td><td className="px-3 py-1.5 text-slate-600">≥ 3.5 kWh/m²/day</td></tr>
                  <tr className="border-t border-slate-100"><td className="px-3 py-1.5 font-medium text-red-700">Fair</td><td className="px-3 py-1.5 text-slate-600">&lt; 3.5 kWh/m²/day</td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div className="space-y-3">
            <p className="font-semibold text-gray-800">Prediction Confidence (35–99%)</p>
            <p className="text-xs text-gray-600">Two separate scores — one per model — both clipped to [35, 99]:</p>
            <div className="rounded-lg border border-blue-200 bg-blue-50 p-2.5 space-y-1 text-xs">
              <p className="font-semibold text-blue-800">Baseline — spatial proximity only</p>
              <p className="font-mono text-blue-700">coverage = 100 − dist_km × 2.5</p>
              <p className="text-blue-700">Baseline is the reference; its confidence reflects only how well the location is covered by training data.</p>
            </div>
            <div className="rounded-lg border border-orange-200 bg-orange-50 p-2.5 space-y-1 text-xs">
              <p className="font-semibold text-orange-800">FI-AdaBoost — spatial + model agreement</p>
              <p className="font-mono text-orange-700">agreement = 100 − |Δirradiance| × 20</p>
              <p className="font-mono text-orange-700">FI confidence = 0.7 × coverage + 0.3 × agreement</p>
              <p className="text-orange-700">Penalised when it diverges strongly from the baseline — a large irradiance gap lowers this score.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Interpreting the results */}
      <Card className="border-2 border-teal-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <TrendingUp className="w-5 h-5 text-teal-600" />
            Interpreting the Results
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <div className="space-y-3">
            {[
              {
                label: '30.7% RMSE reduction (Experiment 1)',
                detail: "FI-AdaBoost's typical irradiance error is ~1,495 J/m²/day smaller than the baseline. Because SEP = irradiance × rooftop area, this directly improves solar energy potential estimates — a building with 100 m² rooftop would see its daily SEP estimate carry ~0.415 kWh/day (~152 kWh/year) less error.",
              },
              {
                label: 'DM statistic = 14.74 (p < 0.001) — Spatial domain',
                detail: 'Extremely strong statistical evidence that the accuracy difference is real, not sampling noise. At n ≈ 3,000, the critical value at α = 0.05 is well below 14.74 — the result would survive even very conservative corrections for multiple testing.',
              },
              {
                label: 'Daily DM = −1.54 (p = 0.124) — Temporal domain',
                detail: "No statistically significant difference on the 365-point daily time series. Both models have similar accuracy on temporal variation — FI-AdaBoost's advantage is specific to the spatial variation task (predicting across different buildings and locations), not to temporal forecasting.",
              },
              {
                label: 'R² close to 1.0',
                detail: 'The model explains nearly all variance in irradiance across buildings. The remaining gap is attributed to unmodelled factors: micro-climate variation below NASA POWER resolution, local shading not captured in OSM, and rooftop surface properties.',
              },
            ].map(({ label, detail }) => (
              <div key={label} className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
                <p className="font-semibold text-gray-800 text-xs uppercase tracking-wide mb-1">{label}</p>
                <p className="text-xs text-gray-600 leading-relaxed">{detail}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Data sources */}
      <Card className="border-2 border-sky-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Database className="w-5 h-5 text-sky-600" />
            Data Sources
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {[
            {
              source: 'NASA POWER API',
              what: 'Annual Global Horizontal Irradiance (GHI) at 3,000 random coordinates — 2024 data, ~50 km resolution grid',
              color: 'blue',
            },
            {
              source: 'OpenStreetMap / Overpass API',
              what: 'Building footprint polygons for Davao City — used to compute azimuth, rooftop area, orientation score, shading factor, and Solar Exposure Index (SEI)',
              color: 'emerald',
            },
          ].map(({ source, what, color }) => (
            <div key={source} className={`rounded-xl border border-${color}-200 bg-${color}-50 p-4`}>
              <p className={`font-semibold text-${color}-800`}>{source}</p>
              <p className={`text-xs text-${color}-700 mt-1`}>{what}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Where to find things */}
      <Card className="border-2 border-amber-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Layers className="w-5 h-5 text-amber-600" />
            Where to Find Things
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {[
            {
              icon: <Sun className="w-5 h-5 text-white" />,
              gradient: 'from-orange-500 to-amber-500',
              tab: 'Forecasting Tool',
              desc: 'Click a location on the map or search an address. The backend fetches live OSM building geometry, computes the 8 features, and returns a GHI prediction and annual solar energy potential for that rooftop.',
            },
            {
              icon: <Map className="w-5 h-5 text-white" />,
              gradient: 'from-blue-500 to-cyan-600',
              tab: 'Location Analysis',
              desc: 'Side-by-side comparison of Baseline AdaBoost and FI-AdaBoost predictions for the selected location. Shows the rooftop features (orientation, shading, SEI) and meteorological context computed from live OSM data.',
            },
            {
              icon: <BarChart3 className="w-5 h-5 text-white" />,
              gradient: 'from-emerald-500 to-teal-600',
              tab: 'Model Analysis',
              desc: 'Training-wide performance metrics from the saved results files. Bar charts for RMSE, MAE, and R². 5-fold spatial CV table. Diebold-Mariano statistical test. Daily temporal split metrics. Research plots from the last training run. Results correspond to whichever branch (experiment) the backend was trained on.',
            },
          ].map(({ icon, gradient, tab, desc }) => (
            <div key={tab} className="flex gap-3 items-start rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className={`w-9 h-9 bg-gradient-to-br ${gradient} rounded-lg flex items-center justify-center shadow shrink-0`}>
                {icon}
              </div>
              <div>
                <p className="font-semibold text-gray-800">{tab}</p>
                <p className="text-xs text-gray-600 mt-1">{desc}</p>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Scope and Limitations */}
      <Card className="border-2 border-rose-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <AlertTriangle className="w-5 h-5 text-rose-600" />
            Scope and Limitations
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-700 leading-relaxed">
          <ul className="space-y-2 list-disc list-inside text-xs text-gray-600">
            <li><strong>No panel efficiency modelled</strong> — SEP represents incident solar energy on the rooftop surface, not usable electricity output. Actual PV yield requires multiplying by panel efficiency (~15–22%) and system performance ratio (~0.75–0.85).</li>
            <li><strong>OSM completeness varies</strong> — Buildings not mapped in OpenStreetMap, or mapped with coarse polygon geometry, will produce less accurate rooftop area, azimuth, and shading estimates. Rural or newly developed areas of Davao City may be under-represented.</li>
            <li><strong>NASA POWER resolution (~50 km)</strong> — The irradiance data does not capture local micro-climate effects such as urban heat islands, coastal sea-breeze patterns, or topographic shading from mountains near Davao Gulf.</li>
            <li><strong>Davao City training domain only</strong> — The model was trained exclusively on coordinates within Davao City. Predictions for locations outside this area are extrapolations; the confidence score will be low but the model may still run without error.</li>
            <li><strong>Rooftop obstructions excluded</strong> — HVAC units, water tanks, parapet walls, and self-shading from adjacent taller buildings are not captured in OSM building footprints and are not accounted for in the SEP estimate.</li>
            <li><strong>Static annual average</strong> — The model predicts an annual average irradiance. Seasonal variation (wet season vs. dry season in Mindanao) is not decomposed in the single SEP output.</li>
          </ul>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <span>Mercado · Retardo · Verzosa &nbsp;·&nbsp; University of Mindanao, College of Computing Studies &nbsp;·&nbsp; Academic Year 2026</span>
      </div>
    </div>
  );
}

