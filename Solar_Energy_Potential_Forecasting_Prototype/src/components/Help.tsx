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
            The pipeline samples 20,000 random coordinates within Davao City, fetches annual solar irradiance from <strong>NASA POWER</strong>, derives rooftop features from <strong>OpenStreetMap</strong> building footprints, and trains both models on the same target: <em>pvlib POA-adjusted effective irradiance (J/m²/day)</em>.
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
            <p className="text-xs text-gray-600">Blended from two signals — both clipped to [35, 99]:</p>
            <ul className="space-y-2 text-xs text-gray-600 list-disc list-inside">
              <li><strong>Coverage:</strong> distance from pin to nearest training point (kd-tree) → <code className="bg-slate-100 px-1 rounded">100 − dist_km × 2.5</code></li>
              <li><strong>Agreement:</strong> model disagreement penalty → <code className="bg-slate-100 px-1 rounded">100 − |Baseline − FI| × 20</code></li>
              <li><strong>Final:</strong> <code className="bg-slate-100 px-1 rounded">0.7 × coverage + 0.3 × agreement</code></li>
            </ul>
            <p className="text-xs text-gray-500 italic">Confidence drops when the pin is far from any training point or when the two models strongly disagree.</p>
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
              what: 'Annual Global Horizontal Irradiance (GHI) at 20,000 random coordinates — 2024 data, ~50 km resolution grid',
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
              desc: 'Training-wide performance metrics from the saved results files. Bar charts for RMSE, MAE, and R². Cross-validation fold table. Results shown here correspond to whichever branch (experiment) the backend was trained on.',
            },
            {
              icon: <FlaskConical className="w-5 h-5 text-white" />,
              gradient: 'from-violet-500 to-fuchsia-600',
              tab: 'Global Analytics',
              desc: 'City-wide solar energy potential summary across all 20,000 sampled buildings. Distribution charts, total annual SEP estimates, and per-building breakdowns comparing both models.',
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

      {/* Footer */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <span>Mercado · Retardo · Verzosa &nbsp;·&nbsp; University of Mindanao, College of Computing Studies &nbsp;·&nbsp; Academic Year 2026</span>
      </div>
    </div>
  );
}

