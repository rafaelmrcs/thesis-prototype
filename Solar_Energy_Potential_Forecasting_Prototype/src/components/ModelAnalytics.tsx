import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { RefreshCw } from 'lucide-react';
import { fetchBackendJson } from '../lib/backend';

interface ModelAnalyticsProps {
  lat?: number;
  lng?: number;
}

interface PredictSnapshot {
  solarPotential: number;
  rooftopArea: number;
  solarExposureIndex: number;
  orientation: string;
  azimuth: number;
  sunshineHours: number;
  cloudCover: number;
  temperature: number;
  humidity: number;
  clearSkyRatio: number;
}

interface ComparisonData {
  baseline: PredictSnapshot;
  fiAdaBoost: PredictSnapshot;
  improvement: {
    solarPotentialDiff: number;
    solarPotentialImprovementPct: number;
    confidenceLevel: number;
  };
  performanceMetricsComparison?: {
    baseline: { rmse: number; mae: number; r2: number };
    fiAdaBoost: { rmse: number; mae: number; r2: number };
    rmseImprovementPct: number;
    maeImprovementPct: number;
    r2ImprovementPct: number;
  };
}

function mapLiveComparisonError(message: string): string {
  if (/Live rooftop lookup is temporarily unavailable because all configured Overpass endpoints failed/i.test(message)) {
    return 'Live OpenStreetMap rooftop lookup is temporarily unavailable right now. Please try refreshing this analysis shortly.';
  }
  return message;
}

export function ModelAnalytics({ lat = 7.0731, lng = 125.6128 }: ModelAnalyticsProps) {
  const [liveComparison, setLiveComparison] = useState<ComparisonData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLiveComparison = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchBackendJson<ComparisonData>('/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng }),
      });
      setLiveComparison(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to compare models';
      setError(mapLiveComparisonError(message));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void fetchLiveComparison();
  }, [lat, lng]);

  return (
    <div className="space-y-6">
      {/* Header banner */}
      <div className="rounded-xl border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm text-cyan-800">
        <span className="font-semibold">Location Analysis</span> — predictions and rooftop context for the
        currently selected pin. Change the pin on the map to refresh.
      </div>

      {error && (
        <Card className="border-2 border-red-200 bg-red-50">
          <CardContent className="pt-6 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {/* Live prediction comparison */}
      <Card className="border-2 border-cyan-200 bg-gradient-to-br from-cyan-50 via-blue-50 to-sky-50 shadow-xl">
        <CardHeader>
          <div className="flex items-center justify-between gap-4">
            <div>
              <CardTitle className="text-xl">Live Model Prediction</CardTitle>
              <CardDescription>
                Both models predicting for {lat.toFixed(4)}°N, {lng.toFixed(4)}°E
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={fetchLiveComparison} disabled={isLoading} className="gap-2">
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {liveComparison ? (
            <>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <div className="rounded-xl border-2 border-slate-200 bg-white p-4 shadow-sm">
                  <p className="text-xs text-slate-500 uppercase tracking-wide">Baseline AdaBoost</p>
                  <p className="mt-1 text-3xl font-semibold text-slate-700">
                    {liveComparison.baseline.solarPotential.toFixed(3)}
                  </p>
                  <p className="text-xs text-slate-400">kWh/m²/day (sky irradiance — geometry-blind)</p>
                </div>
                <div className="rounded-xl border-2 border-orange-200 bg-white p-4 shadow-sm">
                  <p className="text-xs text-slate-500 uppercase tracking-wide">FI-AdaBoost</p>
                  <p className="mt-1 text-3xl font-semibold text-orange-600">
                    {liveComparison.fiAdaBoost.solarPotential.toFixed(3)}
                  </p>
                  <p className="text-xs text-slate-400">kWh/m²/day (geometry-adjusted for this rooftop)</p>
                </div>
                <div className="rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 p-4 text-white shadow-lg">
                  <p className="text-xs opacity-80 uppercase tracking-wide">Difference</p>
                  <p className="mt-1 text-3xl font-semibold">
                    {liveComparison.improvement.solarPotentialDiff >= 0 ? '+' : ''}
                    {liveComparison.improvement.solarPotentialDiff.toFixed(3)}
                  </p>
                  <p className="text-xs opacity-70">
                    ({liveComparison.improvement.solarPotentialImprovementPct >= 0 ? '+' : ''}
                    {liveComparison.improvement.solarPotentialImprovementPct.toFixed(2)}%)
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-sky-200 bg-white p-4 shadow-sm">
                <p className="text-xs text-slate-500 uppercase tracking-wide">Prediction Confidence</p>
                <p className="mt-1 text-3xl font-semibold text-sky-700">
                  {liveComparison.improvement.confidenceLevel.toFixed(1)}%
                </p>
                <p className="text-xs text-slate-400">Based on proximity to training data and model agreement</p>
              </div>

              {liveComparison.performanceMetricsComparison && (
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <p className="text-xs text-slate-500 uppercase tracking-wide">Saved Test Metrics (from results/metrics_summary.csv)</p>
                  <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
                    <div>
                      <p className="text-xs text-slate-500">RMSE (kWh/m²/day)</p>
                      <p className="text-sm text-slate-700">Baseline: {liveComparison.performanceMetricsComparison.baseline.rmse.toFixed(5)}</p>
                      <p className="text-sm text-orange-600">FI: {liveComparison.performanceMetricsComparison.fiAdaBoost.rmse.toFixed(5)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">MAE (kWh/m²/day)</p>
                      <p className="text-sm text-slate-700">Baseline: {liveComparison.performanceMetricsComparison.baseline.mae.toFixed(5)}</p>
                      <p className="text-sm text-orange-600">FI: {liveComparison.performanceMetricsComparison.fiAdaBoost.mae.toFixed(5)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">R²</p>
                      <p className="text-sm text-slate-700">Baseline: {liveComparison.performanceMetricsComparison.baseline.r2.toFixed(4)}</p>
                      <p className="text-sm text-orange-600">FI: {liveComparison.performanceMetricsComparison.fiAdaBoost.r2.toFixed(4)}</p>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white/70 p-6 text-sm text-slate-500">
              {isLoading ? 'Fetching predictions from backend…' : 'No data — check the backend connection.'}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Rooftop & weather context */}
      {liveComparison && (
        <Card className="border-2 border-slate-200 bg-white shadow-xl">
          <CardHeader>
            <CardTitle className="text-xl">Rooftop &amp; Weather Context</CardTitle>
            <CardDescription>
              Engineered feature values for {lat.toFixed(4)}°N, {lng.toFixed(4)}°E
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <SnapshotTile label="Rooftop Area" value={`${liveComparison.fiAdaBoost.rooftopArea.toFixed(1)} m²`} />
              <SnapshotTile label="Solar Exposure Index" value={liveComparison.fiAdaBoost.solarExposureIndex.toFixed(3)} />
              <SnapshotTile label="Sunshine Hours" value={`${liveComparison.fiAdaBoost.sunshineHours.toFixed(1)} hrs/day`} />
              <SnapshotTile label="Cloud Cover" value={`${liveComparison.fiAdaBoost.cloudCover.toFixed(1)}%`} />
              <SnapshotTile label="Temperature" value={`${liveComparison.fiAdaBoost.temperature.toFixed(1)}°C`} />
              <SnapshotTile label="Humidity" value={`${liveComparison.fiAdaBoost.humidity.toFixed(1)}%`} />
              <SnapshotTile label="Orientation" value={liveComparison.fiAdaBoost.orientation} />
              <SnapshotTile label="Azimuth" value={`${liveComparison.fiAdaBoost.azimuth.toFixed(1)}°`} />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function SnapshotTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm">
      <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-900">{value}</p>
    </div>
  );
}
