import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import {
  Sun,
  Map,
  BarChart3,
  FlaskConical,
  Database,
  Cpu,
  Layers,
  ArrowRight,
  Building2,
  Satellite,
  Calculator,
  HelpCircle,
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
              A research prototype that predicts rooftop solar energy potential for buildings in <strong>Davao City, Philippines</strong> using a custom <strong>Feature-Importance-Aware AdaBoost (FI-AdaBoost)</strong> regression model.
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              <Badge className="bg-orange-100 text-orange-700 border border-orange-200">Thesis Prototype</Badge>
              <Badge className="bg-blue-100 text-blue-700 border border-blue-200">OpenStreetMap Live Data</Badge>
              <Badge className="bg-violet-100 text-violet-700 border border-violet-200">ML + Physics Pipeline</Badge>
            </div>
          </div>
        </div>
      </div>

      {/* What this does */}
      <Card className="border-2 border-blue-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <HelpCircle className="w-5 h-5 text-blue-600" />
            What this system does
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <p>
            This system estimates how much solar energy a rooftop at any point in Davao City could generate per year. It works in two phases:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-blue-100 bg-blue-50 p-4">
              <p className="font-semibold text-blue-800 mb-1">Phase 1 — ML Prediction</p>
              <p className="text-blue-700 text-xs">
                A trained FI-AdaBoost model predicts the <strong>Global Horizontal Irradiance (GHI)</strong> at the selected location using the building's geometry features (orientation, shading, solar exposure). GHI is returned in <strong>kWh/m²/day</strong>.
              </p>
            </div>
            <div className="rounded-xl border border-amber-100 bg-amber-50 p-4">
              <p className="font-semibold text-amber-800 mb-1">Phase 2 — Solar Energy Potential Formula</p>
              <p className="text-amber-700 text-xs">
                The predicted GHI is multiplied by rooftop area to estimate <strong>daily solar energy potential in kWh/day</strong>. This is the total solar energy incident on the rooftop surface — a theoretical value before any conversion to electricity. Panel efficiency and system losses are outside the scope of this study.
              </p>
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 font-mono text-xs text-slate-700 space-y-1">
            <p>Daily SEP (kWh/day) = Rooftop Area (m²) × GHI (kWh/m²/day)</p>
            <p>Annual SEP (kWh/year) = Daily SEP × 365</p>
          </div>
        </CardContent>
      </Card>

      {/* How live features work */}
      <Card className="border-2 border-emerald-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Map className="w-5 h-5 text-emerald-600" />
            How live rooftop features are fetched (like Google Sunroof)
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <p>
            Instead of looking up a static table, the backend queries <strong>OpenStreetMap</strong> in real time every time you predict. This is the same principle behind Google Sunroof's building geometry analysis.
          </p>
          <div className="space-y-3">
            {[
              {
                icon: <Satellite className="w-4 h-4 text-emerald-600" />,
                step: '1. Query Overpass API',
                desc: 'Fetches all building footprints within 100 m of the clicked coordinate from OpenStreetMap\'s Overpass API.',
              },
              {
                icon: <Building2 className="w-4 h-4 text-blue-600" />,
                step: '2. Find nearest building',
                desc: 'Selects the building whose geometric centroid is closest to the pin location.',
              },
              {
                icon: <Calculator className="w-4 h-4 text-violet-600" />,
                step: '3. Compute features',
                desc: 'From the polygon nodes: orientation score (bounding-box azimuth → south-facing ratio), rooftop area (shoelace formula in m²), and shading factor (density of nearby buildings within 50 m).',
              },
              {
                icon: <Layers className="w-4 h-4 text-orange-600" />,
                step: '4. Derive Solar Exposure Index (SEI)',
                desc: 'SEI = orientation × area × (1 − shading), normalised by the maximum SEI seen in the training set so it falls in [0, 1].',
              },
              {
                icon: <Cpu className="w-4 h-4 text-red-600" />,
                step: '5. Model inference',
                desc: 'The 5 features [lat, lon, orientation_score, shading_factor, SEI_norm] are fed into the saved FI-AdaBoost model to predict GHI.',
              },
            ].map(({ icon, step, desc }) => (
              <div key={step} className="flex gap-3 items-start">
                <div className="w-7 h-7 rounded-full bg-white border-2 border-slate-200 flex items-center justify-center shrink-0 shadow-sm mt-0.5">
                  {icon}
                </div>
                <div>
                  <p className="font-semibold text-gray-800">{step}</p>
                  <p className="text-gray-600 text-xs">{desc}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-800">
            <strong>Fallback:</strong> If the Overpass API is unreachable (timeout or no internet), the backend silently falls back to the nearest training data point using a KD-tree — so predictions always return without an error.
          </div>
        </CardContent>
      </Card>

      {/* The FI-AdaBoost algorithm */}
      <Card className="border-2 border-violet-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <FlaskConical className="w-5 h-5 text-violet-600" />
            The FI-AdaBoost Algorithm
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm text-gray-700 leading-relaxed">
          <p>
            Standard AdaBoost re-weights training samples after each weak learner — samples that were mispredicted get higher weight so the next learner focuses on them. <strong>FI-AdaBoost extends this</strong> by also modulating sample weights with a <strong>feature importance factor Φᵢ</strong>.
          </p>
          <div className="rounded-xl border border-violet-100 bg-violet-50 p-4 font-mono text-xs text-violet-900 space-y-1">
            <p className="font-sans font-semibold text-violet-800 mb-2">Weight update rule (per boosting round):</p>
            <p>Standard AdaBoost:  w_i ← w_i × exp(α × 𝟙[error_i])</p>
            <p>FI-AdaBoost:        w_i ← w_i × exp(α × 𝟙[error_i] × Φᵢ)</p>
            <p className="font-sans text-violet-700 text-xs mt-2">
              Φᵢ = weighted average of feature importances for sample i, computed from the current ensemble.
              Samples in feature-rich, informative regions get amplified weight → the next learner pays more attention there.
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <p className="font-semibold text-slate-700 mb-1">Baseline AdaBoost</p>
              <p className="text-xs text-slate-600">Features: <code className="bg-slate-100 px-1 rounded">lat, lon</code></p>
              <p className="text-xs text-slate-600 mt-1">Predicts GHI from coordinates only — captures coarse NASA POWER grid zones.</p>
            </div>
            <div className="rounded-xl border border-orange-200 bg-orange-50 p-4 shadow-sm">
              <p className="font-semibold text-orange-800 mb-1">FI-AdaBoost (Proposed)</p>
              <p className="text-xs text-orange-700">Features: <code className="bg-white px-1 rounded border border-orange-200">lat, lon, orientation_score, shading_factor, SEI_norm</code></p>
              <p className="text-xs text-orange-700 mt-1">Predicts GHI with building geometry awareness and custom importance-weighted boosting.</p>
            </div>
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
        <CardContent>
          <div className="space-y-3 text-sm">
            {[
              {
                source: 'NASA POWER API',
                what: 'Global Horizontal Irradiance (GHI) — ~50 km resolution grid, 2024 annual mean',
                note: 'Coarse resolution: ~75% of the 3,000 Davao City points share the same GHI value (4.963 kWh/m²/day). This is a known study limitation.',
                color: 'blue',
              },
              {
                source: 'OpenStreetMap / Overpass API',
                what: 'Building footprint polygons — used live at prediction time to compute orientation, area, and shading',
                note: 'Same data source as the training dataset. Fetched fresh per query.',
                color: 'emerald',
              },
              {
                source: 'Integrated Dataset',
                what: '3,000 buildings across Davao City with lat, lon, rooftop area, orientation, shading factor, SEI, and GHI',
                note: 'Used to train and evaluate both models. 80/20 train/test split.',
                color: 'violet',
              },
            ].map(({ source, what, note, color }) => (
              <div key={source} className={`rounded-xl border border-${color}-200 bg-${color}-50 p-4`}>
                <p className={`font-semibold text-${color}-800`}>{source}</p>
                <p className={`text-xs text-${color}-700 mt-1`}>{what}</p>
                <p className={`text-xs text-${color}-600 mt-1 italic`}>{note}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Guide to tabs */}
      <Card className="border-2 border-amber-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Layers className="w-5 h-5 text-amber-600" />
            Guide to Each Tab
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          {[
            {
              icon: <Sun className="w-5 h-5 text-orange-500" />,
              tab: 'Forecasting Tool',
              gradient: 'from-orange-500 to-amber-500',
              steps: [
                'Type an address or click on the map to place a pin.',
                'Click "Predict Solar Potential" — the backend fetches live OSM building geometry for that exact coordinate.',
                'The FI-AdaBoost model predicts GHI (kWh/m²/day).',
                'Solar Energy Potential is calculated locally: Daily SEP = Area × GHI; Annual SEP = Daily SEP × 365.',
              ],
            },
            {
              icon: <BarChart3 className="w-5 h-5 text-blue-500" />,
              tab: 'Location Analysis',
              gradient: 'from-blue-500 to-cyan-600',
              steps: [
                'Automatically syncs with the pin you set in the Forecasting Tool.',
                'Shows side-by-side GHI predictions from Baseline AdaBoost and FI-AdaBoost.',
                'Displays saved test-set performance metrics (RMSE, MAE, R²) in kWh/m²/day.',
                'Shows the rooftop and weather context features computed from OSM.',
              ],
            },
            {
              icon: <FlaskConical className="w-5 h-5 text-violet-500" />,
              tab: 'Model Analysis',
              gradient: 'from-violet-500 to-fuchsia-600',
              steps: [
                'Shows training-wide performance metrics independent of any selected location.',
                'Bar charts compare RMSE, MAE, and R² between both models.',
                'Error distribution histogram shows residual spread (tighter = better).',
                'Cross-validation fold table (falls back to test-set summary if no CV artifact exists).',
              ],
            },
          ].map(({ icon, tab, gradient, steps }) => (
            <div key={tab} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-8 h-8 bg-gradient-to-br ${gradient} rounded-lg flex items-center justify-center shadow`}>
                  {icon}
                </div>
                <p className="font-semibold text-gray-800">{tab}</p>
              </div>
              <ol className="space-y-1.5">
                {steps.map((step, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                    <span className="w-4 h-4 rounded-full bg-slate-100 text-slate-500 flex items-center justify-center shrink-0 text-[10px] font-bold mt-0.5">
                      {i + 1}
                    </span>
                    {step}
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Glossary / FAQ */}
      <Card className="border-2 border-slate-200 shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <HelpCircle className="w-5 h-5 text-slate-600" />
            Glossary &amp; FAQ
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Accordion type="multiple" className="space-y-2">
            {[
              {
                q: 'What is GHI (Global Horizontal Irradiance)?',
                a: 'GHI is the total solar radiation received per unit area on a horizontal surface, averaged over a day. It is measured in kWh/m²/day and is the primary indicator of how much solar energy is available at a location. A value of ~5 kWh/m²/day is typical for tropical cities like Davao City.',
              },
              {
                q: 'What is SEI (Solar Exposure Index)?',
                a: 'SEI is a composite feature that combines rooftop orientation, area, and shading into a single score. It is computed as: orientation_score × rooftop_area × (1 − shading_factor), then normalised to [0, 1]. A higher SEI means the building is more exposed to direct sunlight.',
              },
              {
                q: 'What does orientation_score mean?',
                a: 'Orientation score measures how "south-facing" a building\'s longest axis is, in the range [0, 1]. A score of 1.0 means the rooftop faces directly south (optimal for solar panels in the Northern Hemisphere). It is computed from the bounding-box azimuth of the OSM building polygon.',
              },
              {
                q: 'What does shading_factor mean?',
                a: 'Shading factor is a proxy for inter-building shade. It counts how many other buildings are within 50 m of the target building and converts that density to a [0, 1] score. A higher value means more potential shading from surrounding structures.',
              },
              {
                q: 'Why are RMSE and MAE so small?',
                a: 'RMSE and MAE are reported in kWh/m²/day. Because GHI barely varies across Davao City (75% of training points share the same NASA POWER grid cell value of 4.963 kWh/m²/day), errors are small in absolute terms but the high R² is partly misleading — the model is mostly learning which coarse grid cell a coordinate belongs to, not fine-grained spatial variation.',
              },
              {
                q: 'Why does FI-AdaBoost sometimes score lower R² than Baseline?',
                a: 'After fixing the methodology (both models now predict the same raw GHI target), FI-AdaBoost (R²≈0.925) is actually slightly worse than Baseline (R²≈0.963) at predicting GHI — because GHI is a sky property, not a building property. The building features (orientation, shading) do not strongly correlate with irradiance at this spatial resolution. FI-AdaBoost\'s contribution is in the Phase 2 energy estimate, which uses building-specific corrections.',
              },
              {
                q: 'What does confidence level mean in Location Analysis?',
                a: 'Confidence is a composite score (0–100%) based on: proximity to the nearest training data point, density of training data around the query location, and how closely the two model predictions agree. It is not a statistical confidence interval — think of it as a "how well-covered is this area by training data" indicator.',
              },
              {
                q: 'What happens if OpenStreetMap has no building at my pin?',
                a: 'If the Overpass API returns no buildings within 100 m, the backend silently falls back to the nearest training dataset point using a KD-tree spatial index. The prediction will still return — the feature values will come from the training record geographically closest to your pin.',
              },
              {
                q: 'Why is Solar Energy Potential different from actual electricity output?',
                a: 'Solar Energy Potential (SEP) in this study represents the total solar energy incident on the rooftop surface — calculated as GHI × Rooftop Area. It is a theoretical value before any energy conversion. Actual electricity output depends on panel efficiency, inverter losses, wiring, shading, and other installation-specific factors, which are outside the scope of this study.',
              },
            ].map(({ q, a }, i) => (
              <AccordionItem key={i} value={`item-${i}`} className="border border-slate-200 rounded-xl px-4 bg-white shadow-sm">
                <AccordionTrigger className="text-sm font-medium text-gray-800 hover:no-underline py-3">
                  <span className="flex items-center gap-2 text-left">
                    <ArrowRight className="w-3 h-3 text-orange-500 shrink-0" />
                    {q}
                  </span>
                </AccordionTrigger>
                <AccordionContent className="text-xs text-gray-600 leading-relaxed pb-3">
                  {a}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </CardContent>
      </Card>

      {/* Footer note */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500 text-center">
        Thesis prototype — University of Mindanao, College of Computing Studies &nbsp;·&nbsp; Davao City solar resource data via NASA POWER &nbsp;·&nbsp; Building geometry via OpenStreetMap
      </div>
    </div>
  );
}
