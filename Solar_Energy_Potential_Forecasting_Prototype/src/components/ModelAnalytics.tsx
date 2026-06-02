import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { RefreshCw } from 'lucide-react';
import { fetchBackendJson } from '../lib/backend';
import { formatExact, formatResultsErrorMetric, formatRawR2, formatPercent } from '../lib/formatting';

interface ModelAnalyticsProps {
  lat?: number;
  lng?: number;
}

interface PredictSnapshot {
  solarPotential: number;
  predictedIrradiance: number;
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
    baselineConfidenceLevel: number;
    fiConfidenceLevel: number;
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
                  <p className="text-xs text-slate-500 uppercase tracking-wide">Baseline AdaBoost SEP</p>
                  <p className="mt-1 text-3xl font-semibold text-slate-700">
                    {formatExact(liveComparison.baseline.solarPotential, 2)}
                  </p>
                  <p className="text-xs text-slate-400">kWh/day theoretical rooftop energy</p>
                  <p className="mt-2 text-xs text-slate-500">
                    Irradiance: {formatExact(liveComparison.baseline.predictedIrradiance, 4)} kWh/m²/day
                  </p>
                </div>
                <div className="rounded-xl border-2 border-orange-200 bg-white p-4 shadow-sm">
                  <p className="text-xs text-slate-500 uppercase tracking-wide">FI-AdaBoost SEP</p>
                  <p className="mt-1 text-3xl font-semibold text-orange-600">
                    {formatExact(liveComparison.fiAdaBoost.solarPotential, 2)}
                  </p>
                  <p className="text-xs text-slate-400">kWh/day theoretical rooftop energy</p>
                  <p className="mt-2 text-xs text-slate-500">
                    Irradiance: {formatExact(liveComparison.fiAdaBoost.predictedIrradiance, 4)} kWh/m²/day
                  </p>
                </div>
              </div>

          
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
              <SnapshotTile label="Rooftop Area" value={`${formatExact(liveComparison.fiAdaBoost.rooftopArea, 1)} m²`} />
              <SnapshotTile label="Solar Exposure Index" value={formatExact(liveComparison.fiAdaBoost.solarExposureIndex, 6)} />
              <SnapshotTile label="Sunshine Hours" value={`${formatExact(liveComparison.fiAdaBoost.sunshineHours, 1)} hrs/day`} />
              <SnapshotTile label="Cloud Cover" value={`${formatPercent(liveComparison.fiAdaBoost.cloudCover, 1)}`} />
              <SnapshotTile label="Temperature" value={`${formatExact(liveComparison.fiAdaBoost.temperature, 1)}°C`} />
              <SnapshotTile label="Humidity" value={`${formatPercent(liveComparison.fiAdaBoost.humidity, 1)}`} />
              <SnapshotTile label="Orientation" value={liveComparison.fiAdaBoost.orientation} />
              <SnapshotTile label="Azimuth" value={`${formatExact(liveComparison.fiAdaBoost.azimuth, 1)}°`} />
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
